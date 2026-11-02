"""The data pipeline's output, pinned to main 72ebfd1 (course finding N3).

Round 1 rewrote the preprocessing, labeling and release-building code. These tests run the current code on a
fixed raw tree and fixed text, and compare every output with what main 72ebfd1 produced from the same input:
- each file of a release built twice from a small SponSpeech-like raw tree (without and with a titles snapshot);
- normalize, the speaker inventory, the video context, tokenize and the punctuation, casing and joint labels.

How the EXPECTED_* values were made: this module's own helpers were run with PYTHONPATH set to the src/ of a
72ebfd1 checkout (RECAP_golden/main_tree), and their output was pasted in below. Never regenerate them from the
current code: a pin taken from the code under test proves nothing.

Three release files depend on the code or the interpreter rather than the data: release_manifest.json records
`code_sha256` (the source fingerprint of the data package, which changed when the code moved) and the Python
version, RELEASE_NOTES.md repeats the fingerprint, and the processed SHA256SUMS lists the digests of both. Those
values are masked in those three files; every other byte of every file is compared. The raw tree is written with
LF line endings and its manifest is sorted by POSIX path, so the input bytes are the same on every platform.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

import pytest

from punct_restore.data import build_release, labeling, load_label_sets, preprocessing

# ---- the fixed input ----

# A small SponSpeech-like raw tree: four source splits; one video seen in two splits; a dev line from another video
# with the same text as a train line, and one with the same words but other marks (the leakage checks); speaker
# labels, a paren tag, an inverted mark folded into '?', an ellipsis and a turn dash; a line that is only a tag
# (rejected), a folded pound sign ('PS10') and a leading minus.
RAW: dict[str, dict[str, str]] = {
    "train": {
        "aaaaaaaaaaa-0": "JAKE: Hello, everyone. Welcome to the show!",
        "aaaaaaaaaaa-1": "We talk about the U.S. economy; it is big -- really big.",
        "aaaaaaaaaaa-2": "Is NetApp really?! (Laughs) Okay.",
        "bbbbbbbbbbb-0": 'In our draft we say "?Es suyo?" here.',
        "bbbbbbbbbbb-1": "Dr. Smith said: we meet at 5 p.m., right? - Yes, e.g. on Monday.",
        "bbbbbbbbbbb-2": "[music]",
        "bbbbbbbbbbb-3": "It costs PS10 now, and the drop is -12 dB.",
    },
    "dev": {
        "aaaaaaaaaaa-0": "Is this the same video as in train?",
        "ddddddddddd-0": "A brand new video, right?",
        "ddddddddddd-1": "So [? vitals ?] were okay... I think.",
        "fffffffffff-0": "Is NetApp really?! (Laughs) Okay.",
        "ggggggggggg-0": "We talk about the U.S. economy, it is big... really big!",
    },
    "test": {
        "ccccccccccc-0": "The FDA met on Monday, I think.",
        "ccccccccccc-1": "[music] Well, it's 3.5% of $10,000!",
    },
    "test-amb": {"ccccccccccc-0": "Same video, different split... okay.", "eeeeeeeeeee-0": "Something else entirely."},
}
TITLES = [
    {"video_id": "aaaaaaaaaaa", "status": 200, "title": "Some Topic | Wikipedia audio article"},
    {"video_id": "ddddddddddd", "status": 200, "title": "A talk about vitals"},
]
LICENSE = "LICENSE\nCC BY-NC 4.0\n"

# Text for the text pipeline: the raw tree's lines plus one line for each rule the release lines do not reach.
EXTRA_TEXTS = [
    "Ben: so we start. Ben: then we stop. Ben: and that is it.",
    "Ben [00:20:58] talks about it at 00:01:02.",
    "Speaker 1: hi there >> how are you",
    "[Student] Can I ask? (whispering) (inaudible) okay",
    "OK JASON MILLER: we go now. ONE: Washing dishes.",
    "ES:He said yes. *sigh* Hebdom?konta is a word.",
    "Visit www.example.com or mail me at a.b@example.org, then -12 dB.",
    "Prof. Lee et al. found No. 5 was no. U.S.A. too.",
    "It is a plant- based diet... right... and I just- I mean -- yes.",
    "Café, naïve résumé; NASA's iPhone and McDonald's.",
    "He asked, \"Really?\" She said: 'yes'!",
    "Wait... what?! Oh, okay.",
    "Hmm.. okay. . . fine, I guess",
    "I think so...",
    "Is it...? Maybe so.",
    "We moved to the U.S.",
    "He was in the U.S. The next day came.",
    "Michael J. Fox spoke here.",
    "THIS WHOLE LINE IS WRITTEN IN CAPITAL LETTERS TODAY.",
    "We saw it (on the news) yesterday.",
    "Okay. Maria: we are ready.",
    "Dr. Kalu [00:20:58] said hi.",
    "Dr. Kalu [00:21:10] went on.",
    "Dr. Kalu [00:22:00] ended it.",
    "Die Straße is long.",  # the tokenizer reads ASCII words, so 'ß' splits the word
    "We cook, bake, etc... and more.",
    "well [...] okay",
    "see .net now",
]
# A comma-free video of more than 300 words: the low-comma (ASR-like) source flag.
LOW_COMMA_VIDEO = [("we keep talking about the same thing " * 50).strip() + "."]


def words_with_commas(count: int, commas: int) -> str:
    """`count` words with a comma after each of the first `commas` hundreds: the low-comma rule's boundaries."""
    words = [f"w{i}" + ("," if i % 100 == 99 and i // 100 < commas else "") for i in range(count)]
    return " ".join(words) + "."


# 300 words with 2 commas is low-comma; 3 commas is exactly the 1.0-per-100-words limit; 299 words is too short.
LOW_COMMA_BOUNDARIES = [words_with_commas(300, 2), words_with_commas(300, 3), words_with_commas(299, 0)]
TTS_TITLE_VARIANT = "Topic | wikipedia AUDIO article "  # another case and a trailing space
# Surfaces where str.lower() and str.casefold() differ: model tokens are lower(), never casefold().
MODEL_TOKEN_SURFACES = ["Straße", "İstanbul", "ΣΊΣΥΦΟΣ", "NASA", "McDonald's"]
SCHEMES = ("extended", "native")

# ---- expected values, computed with main 72ebfd1 (see the module docstring) ----

EXPECTED_RELEASE_FILES: dict[str, str] = {
    "plain/processed/fixture/audit/normalization.jsonl":
        "52990576288ab011be40d7c77a2f338966111a646a4e47f4a256139e25bceb32",
    "plain/processed/fixture/audit/rejected.jsonl": "d072997d0108949923e393748ab7644d4cf091138ee616b23a9fe50cdf32fc1e",
    "plain/processed/fixture/audit/unresolved.jsonl":
        "2d056a2018b8d94bb5cb18507afdd6d15904c35fc8aeee2bf38619d23bbe2320",
    "plain/processed/fixture/benchmark_native/mode_A/test.jsonl":
        "74f7db9cc9350795d270de68fd22e449b5fc9b15bcbb6af9c63ca055ca722ca0",
    "plain/processed/fixture/benchmark_native/mode_A/test_amb.jsonl":
        "6cbd58785ff24716b6abf9a395faa8d063ef857f8b143efe454b5529a246f435",
    "plain/processed/fixture/benchmark_native/mode_A/train.jsonl":
        "7c70826f31cfba9be2896e25fdec9c75d48597fbd1c9946b3739b87bea7bc6d3",
    "plain/processed/fixture/benchmark_native/mode_A/validation.jsonl":
        "8f81eed39cd34857cc8e94f82026e06a692f6419b2d66ef87e0a07bb6b4046af",
    "plain/processed/fixture/masks/case_joint/test.jsonl":
        "f297e281ec70c6a2878eb42f398bc052144579641f67df6d0cf58c4448db1fe7",
    "plain/processed/fixture/masks/case_joint/test_amb.jsonl":
        "56a22d0124ad46acd354f34ffc9c2928e159932c7852a332c6433eb9dfbb1359",
    "plain/processed/fixture/masks/case_joint/train.jsonl":
        "6ae0e6e20da3151e37605d5f31a3d866f02013694397b318a4f63595b1ab1411",
    "plain/processed/fixture/masks/case_joint/validation.jsonl":
        "e1462f8d1ddb607e11a5af468760b6c46f770afe6c74020b776275e5f1f17a82",
    "plain/processed/fixture/mode_A/test.jsonl": "78bcc60a67710342fe02f825fcfd22fcfe8eec8d50ba0ff2922cf545c578b011",
    "plain/processed/fixture/mode_A/test_amb.jsonl": "ab1243c10af58a469b9f6299fc30e364af247959c4a107cfa6e77348bb106d69",
    "plain/processed/fixture/mode_A/train.jsonl": "c398cd097158972047e535ac4cc5046049c9dde72daf3d339bd92bcf6d084a91",
    "plain/processed/fixture/mode_A/validation.jsonl":
        "9c7a4cc5154e8552fef0cbc7164b80c7eb85f4a6add7b10e0ddb0029d9b50fb4",
    "plain/processed/fixture/mode_B/test.jsonl": "9adfbb85e12d10ccac05fbfa79991f9c3db741a216b39b065419d6868d1adf99",
    "plain/processed/fixture/mode_B/test_amb.jsonl": "f6c11cf7c70fd0ecaea869a7bb9a7d1310df904d3c68a71c426427ad086add34",
    "plain/processed/fixture/mode_B/train.jsonl": "9e0751332c1214b914365b50736a3029896e37c28d94aeeb02dd265fbd77cfcf",
    "plain/processed/fixture/mode_B/validation.jsonl":
        "07fd6a19bc3899317c587f0531ea75e9d91f9ad4b65dd92d00ba2d187c97f6c2",
    "plain/processed/fixture/mode_C/test.jsonl": "111597a2b58f3da7f6273d898711b03f3f92a240d50bc5c7d1877d2daddca500",
    "plain/processed/fixture/mode_C/test_amb.jsonl": "b2448d777335ea957119b170c55bed604542cec4fd3075a27726a183dff23076",
    "plain/processed/fixture/mode_C/train.jsonl": "768f649e52f62b9bbc615891db29c01cba199781c29ecdc024636c6af0ff5c38",
    "plain/processed/fixture/mode_C/validation.jsonl":
        "591cdd015e841e27d0a7ec6c1de82c0c06802a4b334b720b2bf50390b7147dca",
    "plain/processed/fixture/release_manifest.json": "e48077cc6a1b5a79bb285b6d808c109dc19187e30f9019ec64b73efe78f34fbd",
    "plain/processed/fixture/RELEASE_NOTES.md": "606cb636ccb2215762556bfcd66bcae5159d41aa553ec49c37658b2eeb7adbbf",
    "plain/processed/fixture/SHA256SUMS": "7fbb96e30fd5631e79bb2ea8ef846860e4c1e32ae02c446966272e2151a8a9f7",
    "plain/processed/fixture/SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt":
        "6d433aaf25060882342d4a0ad6de66b1bb46ac316b5d6d493e50b8130a62d822",
    "plain/processed/fixture/stats.json": "348832aa8a7060b15b282c1feaf7eccd663a58a84475bcca2e99c2025c9e1e36",
    "plain/splits/fixture/default_train/mode_A.ids": "58b675e6d3e54e8c99eab78e1ca9c88308f935eb731f1ede599f611c7931e484",
    "plain/splits/fixture/default_train/mode_B.ids": "58b675e6d3e54e8c99eab78e1ca9c88308f935eb731f1ede599f611c7931e484",
    "plain/splits/fixture/default_train/mode_C.ids": "58b675e6d3e54e8c99eab78e1ca9c88308f935eb731f1ede599f611c7931e484",
    "plain/splits/fixture/leakage_controlled/report.json":
        "16f7d2957edea788ca05b9bc23e5ff74a9f1a38e32df1869890e8556a7f4cd53",
    "plain/splits/fixture/leakage_controlled/test.ids":
        "7ae34c7cd94dc17ec6ad7c5babd21f9b52f894b913a91bba9a645e6a1df8a9a3",
    "plain/splits/fixture/leakage_controlled/test_amb.ids":
        "780cdf255da1d3d33dcc58ca8eb40c1e032d1f5016eeaa6bdd0bfdd8c70b8c37",
    "plain/splits/fixture/leakage_controlled/test_amb_excluded.tsv":
        "b37cd6f4be08f6de98bb4da6057b70b687d81be5785d0805869fc526165c9153",
    "plain/splits/fixture/leakage_controlled/test_excluded.tsv":
        "b37cd6f4be08f6de98bb4da6057b70b687d81be5785d0805869fc526165c9153",
    "plain/splits/fixture/leakage_controlled/validation.ids":
        "134816dac42f78b6f1d8520ffee27c1cbfb71463b3ad0b09bbf1048386442201",
    "plain/splits/fixture/leakage_controlled/validation_excluded.tsv":
        "ea43af2ebc773fb5753e27544d19599824d14da32234dc082afb2a5fead01df5",
    "plain/splits/fixture/primary_eval/test_amb_mode_A.ids":
        "780cdf255da1d3d33dcc58ca8eb40c1e032d1f5016eeaa6bdd0bfdd8c70b8c37",
    "plain/splits/fixture/primary_eval/test_amb_mode_B.ids":
        "780cdf255da1d3d33dcc58ca8eb40c1e032d1f5016eeaa6bdd0bfdd8c70b8c37",
    "plain/splits/fixture/primary_eval/test_amb_mode_C.ids":
        "780cdf255da1d3d33dcc58ca8eb40c1e032d1f5016eeaa6bdd0bfdd8c70b8c37",
    "plain/splits/fixture/primary_eval/test_mode_A.ids":
        "7ae34c7cd94dc17ec6ad7c5babd21f9b52f894b913a91bba9a645e6a1df8a9a3",
    "plain/splits/fixture/primary_eval/test_mode_B.ids":
        "7ae34c7cd94dc17ec6ad7c5babd21f9b52f894b913a91bba9a645e6a1df8a9a3",
    "plain/splits/fixture/primary_eval/test_mode_C.ids":
        "7ae34c7cd94dc17ec6ad7c5babd21f9b52f894b913a91bba9a645e6a1df8a9a3",
    "plain/splits/fixture/primary_eval/validation_mode_A.ids":
        "705a67bf72344364b6aae22ea168cf3b3c3891039d2f40a1bc27f19ea5568de0",
    "plain/splits/fixture/primary_eval/validation_mode_B.ids":
        "705a67bf72344364b6aae22ea168cf3b3c3891039d2f40a1bc27f19ea5568de0",
    "plain/splits/fixture/primary_eval/validation_mode_C.ids":
        "705a67bf72344364b6aae22ea168cf3b3c3891039d2f40a1bc27f19ea5568de0",
    "plain/splits/fixture/SHA256SUMS": "dd88ddee41cd22d574a13351e2de84d5f9ce3179c892db849e071f7004161d18",
    "plain/splits/fixture/split_map.tsv": "71f6171575962954be1b2b3fcad8564046a2c5c51d3ea958574d466a888ea6d9",
    "titled/processed/fixture/audit/normalization.jsonl":
        "52990576288ab011be40d7c77a2f338966111a646a4e47f4a256139e25bceb32",
    "titled/processed/fixture/audit/rejected.jsonl": "d072997d0108949923e393748ab7644d4cf091138ee616b23a9fe50cdf32fc1e",
    "titled/processed/fixture/audit/unresolved.jsonl":
        "d14179ceb4f26ae0d9be87a307abfbdf8a3695f1e2bec358226c95d7a21b1a1b",
    "titled/processed/fixture/benchmark_native/mode_A/test.jsonl":
        "74f7db9cc9350795d270de68fd22e449b5fc9b15bcbb6af9c63ca055ca722ca0",
    "titled/processed/fixture/benchmark_native/mode_A/test_amb.jsonl":
        "6cbd58785ff24716b6abf9a395faa8d063ef857f8b143efe454b5529a246f435",
    "titled/processed/fixture/benchmark_native/mode_A/train.jsonl":
        "7c70826f31cfba9be2896e25fdec9c75d48597fbd1c9946b3739b87bea7bc6d3",
    "titled/processed/fixture/benchmark_native/mode_A/validation.jsonl":
        "8f81eed39cd34857cc8e94f82026e06a692f6419b2d66ef87e0a07bb6b4046af",
    "titled/processed/fixture/masks/case_joint/test.jsonl":
        "f297e281ec70c6a2878eb42f398bc052144579641f67df6d0cf58c4448db1fe7",
    "titled/processed/fixture/masks/case_joint/test_amb.jsonl":
        "56a22d0124ad46acd354f34ffc9c2928e159932c7852a332c6433eb9dfbb1359",
    "titled/processed/fixture/masks/case_joint/train.jsonl":
        "6ae0e6e20da3151e37605d5f31a3d866f02013694397b318a4f63595b1ab1411",
    "titled/processed/fixture/masks/case_joint/validation.jsonl":
        "e1462f8d1ddb607e11a5af468760b6c46f770afe6c74020b776275e5f1f17a82",
    "titled/processed/fixture/mode_A/test.jsonl": "78bcc60a67710342fe02f825fcfd22fcfe8eec8d50ba0ff2922cf545c578b011",
    "titled/processed/fixture/mode_A/test_amb.jsonl":
        "ab1243c10af58a469b9f6299fc30e364af247959c4a107cfa6e77348bb106d69",
    "titled/processed/fixture/mode_A/train.jsonl": "c398cd097158972047e535ac4cc5046049c9dde72daf3d339bd92bcf6d084a91",
    "titled/processed/fixture/mode_A/validation.jsonl":
        "9c7a4cc5154e8552fef0cbc7164b80c7eb85f4a6add7b10e0ddb0029d9b50fb4",
    "titled/processed/fixture/mode_B/test.jsonl": "9adfbb85e12d10ccac05fbfa79991f9c3db741a216b39b065419d6868d1adf99",
    "titled/processed/fixture/mode_B/test_amb.jsonl":
        "f6c11cf7c70fd0ecaea869a7bb9a7d1310df904d3c68a71c426427ad086add34",
    "titled/processed/fixture/mode_B/train.jsonl": "9e0751332c1214b914365b50736a3029896e37c28d94aeeb02dd265fbd77cfcf",
    "titled/processed/fixture/mode_B/validation.jsonl":
        "07fd6a19bc3899317c587f0531ea75e9d91f9ad4b65dd92d00ba2d187c97f6c2",
    "titled/processed/fixture/mode_C/test.jsonl": "111597a2b58f3da7f6273d898711b03f3f92a240d50bc5c7d1877d2daddca500",
    "titled/processed/fixture/mode_C/test_amb.jsonl":
        "b2448d777335ea957119b170c55bed604542cec4fd3075a27726a183dff23076",
    "titled/processed/fixture/mode_C/train.jsonl": "768f649e52f62b9bbc615891db29c01cba199781c29ecdc024636c6af0ff5c38",
    "titled/processed/fixture/mode_C/validation.jsonl":
        "591cdd015e841e27d0a7ec6c1de82c0c06802a4b334b720b2bf50390b7147dca",
    "titled/processed/fixture/release_manifest.json":
        "ea8047605dd4e8cf68e46511da5911ffb22878a51c941f649daa613a8f713fda",
    "titled/processed/fixture/RELEASE_NOTES.md": "8e7c11e4c4b40f98648b6302c9a1c64a8d643c5ab02d2ef2f10be94d49be4c94",
    "titled/processed/fixture/SHA256SUMS": "ca01dd5ebe3316bf0aa90099794b7e8a19a5843c17193ea5d2263491d1e81add",
    "titled/processed/fixture/SPONSPEECH_LICENSE_AND_ATTRIBUTION.txt":
        "6d433aaf25060882342d4a0ad6de66b1bb46ac316b5d6d493e50b8130a62d822",
    "titled/processed/fixture/stats.json": "4e6294d227b2246d222d52262e0ae9c7cae7f5b2c771c80ab79c7a5634327b6f",
    "titled/splits/fixture/default_train/mode_A.ids":
        "1f0eac9a4a6d7111781a1d1fe94465ea56a91fa359a117e9f114cb30113f36ed",
    "titled/splits/fixture/default_train/mode_B.ids":
        "1f0eac9a4a6d7111781a1d1fe94465ea56a91fa359a117e9f114cb30113f36ed",
    "titled/splits/fixture/default_train/mode_C.ids":
        "1f0eac9a4a6d7111781a1d1fe94465ea56a91fa359a117e9f114cb30113f36ed",
    "titled/splits/fixture/leakage_controlled/report.json":
        "16f7d2957edea788ca05b9bc23e5ff74a9f1a38e32df1869890e8556a7f4cd53",
    "titled/splits/fixture/leakage_controlled/test.ids":
        "7ae34c7cd94dc17ec6ad7c5babd21f9b52f894b913a91bba9a645e6a1df8a9a3",
    "titled/splits/fixture/leakage_controlled/test_amb.ids":
        "780cdf255da1d3d33dcc58ca8eb40c1e032d1f5016eeaa6bdd0bfdd8c70b8c37",
    "titled/splits/fixture/leakage_controlled/test_amb_excluded.tsv":
        "b37cd6f4be08f6de98bb4da6057b70b687d81be5785d0805869fc526165c9153",
    "titled/splits/fixture/leakage_controlled/test_excluded.tsv":
        "b37cd6f4be08f6de98bb4da6057b70b687d81be5785d0805869fc526165c9153",
    "titled/splits/fixture/leakage_controlled/validation.ids":
        "134816dac42f78b6f1d8520ffee27c1cbfb71463b3ad0b09bbf1048386442201",
    "titled/splits/fixture/leakage_controlled/validation_excluded.tsv":
        "ea43af2ebc773fb5753e27544d19599824d14da32234dc082afb2a5fead01df5",
    "titled/splits/fixture/primary_eval/test_amb_mode_A.ids":
        "780cdf255da1d3d33dcc58ca8eb40c1e032d1f5016eeaa6bdd0bfdd8c70b8c37",
    "titled/splits/fixture/primary_eval/test_amb_mode_B.ids":
        "780cdf255da1d3d33dcc58ca8eb40c1e032d1f5016eeaa6bdd0bfdd8c70b8c37",
    "titled/splits/fixture/primary_eval/test_amb_mode_C.ids":
        "780cdf255da1d3d33dcc58ca8eb40c1e032d1f5016eeaa6bdd0bfdd8c70b8c37",
    "titled/splits/fixture/primary_eval/test_mode_A.ids":
        "7ae34c7cd94dc17ec6ad7c5babd21f9b52f894b913a91bba9a645e6a1df8a9a3",
    "titled/splits/fixture/primary_eval/test_mode_B.ids":
        "7ae34c7cd94dc17ec6ad7c5babd21f9b52f894b913a91bba9a645e6a1df8a9a3",
    "titled/splits/fixture/primary_eval/test_mode_C.ids":
        "7ae34c7cd94dc17ec6ad7c5babd21f9b52f894b913a91bba9a645e6a1df8a9a3",
    "titled/splits/fixture/primary_eval/validation_mode_A.ids":
        "705a67bf72344364b6aae22ea168cf3b3c3891039d2f40a1bc27f19ea5568de0",
    "titled/splits/fixture/primary_eval/validation_mode_B.ids":
        "705a67bf72344364b6aae22ea168cf3b3c3891039d2f40a1bc27f19ea5568de0",
    "titled/splits/fixture/primary_eval/validation_mode_C.ids":
        "705a67bf72344364b6aae22ea168cf3b3c3891039d2f40a1bc27f19ea5568de0",
    "titled/splits/fixture/SHA256SUMS": "c7076971fb79cb7c15ae5cb564d1836e451457358aafce6e6d24ac6af7c0c0d3",
    "titled/splits/fixture/split_map.tsv": "acdf922eeeb6f9dfd900d75f5747ffe9c8ea0e87daebe7246989efdb42d9c5a8",
}
EXPECTED_TEXT_PIPELINE: dict[str, str] = {
    "speaker_inventory": "7905133140ba00a03696cd487612ed83922885c743f986cc790af29bbfb51ec5",
    "video_context": "3c3a85dd5d21be1a36169eddaa6e282883a5e55c5ac7d974294646342ca9f28c",
    "video_context_low_comma": "9a072fe0064b0c5e16aaaa0d799b7aceafb8368f14b5df0717bec79ee85632f1",
    "video_context_low_comma_boundaries": "394e8688043d4f06806888bf62c0ba99e692a1cd2dbb93170b21e3917b1b346c",
    "video_context_tts_title_variant": "3c3a85dd5d21be1a36169eddaa6e282883a5e55c5ac7d974294646342ca9f28c",
    "normalize": "d2994a3354feb3abe6fe0e9c4e009bf2a35956e4b23c282f6e2926348f59c8a4",
    "tokenize": "557b628262d000b5d998931270d7971ac93ad9908507b99ba11e47335e869035",
    "model_tokens": "3bb509f111e41c60325f141f5b70b1eb4556c727ba3ef14eadcc62228446c597",
    "model_token_surfaces": "30896c49148aa05044fab9066a88e01adf000102deaba0099e6f3e112de664fe",
    "punctuation_extended": "f7f5a89bf32fc06965c76e108890d64fe63e2f3c54df6a4cf1c4c8762d8ba5b1",
    "punctuation_native": "3fa040e474e6ec5083fbc706474982b72c23568b50f4e2e88c42ff13b0a87b85",
    "projected_standard": "01f41346c64b7f3a2fc0c042761dfd9237da49437ac28044c92db9f57c942609",
    "casing": "4edf1d60166bdf4a4ede72038b6475c39ac95802cbe4f26ec29f1920255d62c8",
    "joint": "5280f737704af9e26d4807b219de5f8977fa2b22b2209c07e5e8fe4f5ed3a04a",
    "cross_check_standard": "273af267dded50024d7fdac6bea906d317f4b474b9c1e94286f07aeb03b2d50f",
}


# ---- helpers (also used, unchanged, to compute the expected values on 72ebfd1) ----

FINGERPRINTED = ("RELEASE_NOTES.md", "release_manifest.json")  # the files that record code_sha256
PROCESSED = "processed/fixture"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    text = json.dumps(as_json_data(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(text.encode("utf-8"))


def as_json_data(value: Any) -> Any:
    """`value` as JSON data: dataclasses and tuples become lists of their fields, sets sorted lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [as_json_data(getattr(value, f.name)) for f in dataclasses.fields(value)]
    if isinstance(value, dict):
        return {str(key): as_json_data(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(as_json_data(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [as_json_data(item) for item in value]
    return value


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def write_raw_tree(root: Path) -> Path:
    """The raw tree in SponSpeech's layout, with its MANIFEST.sha256; returns the manifest path."""
    for split, utterances in RAW.items():
        write_lf(root / split / "utt2spk", "".join(f"{u} {u.rsplit('-', 1)[0]}\n" for u in utterances))
        for utterance, text in utterances.items():
            write_lf(root / split / "text" / f"{utterance}.txt", text)
    write_lf(root / "README.txt", LICENSE)
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())  # same order everywhere
    lines = "".join(f"{sha256_bytes((root / name).read_bytes())}  {name}\n" for name in names)
    write_lf(root / "MANIFEST.sha256", lines)
    return root / "MANIFEST.sha256"


def build_fixture_releases(base: Path) -> dict[str, Path]:
    """Build the release twice, as main's call form does: without and with the titles snapshot."""
    manifest = write_raw_tree(base / "raw")
    write_lf(base / "titles.jsonl", "".join(json.dumps(row) + "\n" for row in TITLES))
    builds = {}
    for name, titles in (("plain", None), ("titled", str(base / "titles.jsonl"))):
        out = base / name
        build_release(str(base / "raw"), str(out / "processed"), str(out / "splits"), "fixture",
                      raw_manifest=str(manifest), titles_path=titles, log=lambda _line: None)
        builds[name] = out
    return builds


def release_digests(name: str, out: Path) -> dict[str, str]:
    """The SHA-256 of every file a build wrote, keyed `<build>/<path>`; fingerprint and Python version masked."""
    code_sha = json.loads((out / PROCESSED / "release_manifest.json").read_text(encoding="utf-8"))["code_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", code_sha), code_sha  # a short value would mask unrelated text
    masks = {code_sha: "<code_sha256>", f'"python": "{platform.python_version()}"': '"python": "<python version>"'}
    for fingerprinted in FINGERPRINTED:  # their own digests change with the fingerprint too
        masks[sha256_bytes((out / PROCESSED / fingerprinted).read_bytes())] = f"<sha256 of {fingerprinted}>"
    digests = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        data = path.read_bytes()
        if path.parent == out / PROCESSED and path.name in (*FINGERPRINTED, "SHA256SUMS"):
            text = data.decode("utf-8")
            for value, placeholder in masks.items():
                text = text.replace(value, placeholder)
            data = text.encode("utf-8")
        digests[f"{name}/{path.relative_to(out).as_posix()}"] = sha256_bytes(data)
    return digests


def all_texts() -> list[str]:
    return [text for utterances in RAW.values() for text in utterances.values()] + EXTRA_TEXTS


def text_pipeline_stages() -> dict[str, Any]:
    """Every stage of the text pipeline on the fixture text, with the speaker inventory of all of it as one video."""
    texts = all_texts()
    speakers = preprocessing.build_speaker_inventory(texts)
    normalized = [preprocessing.normalize(text, speakers) for text in texts]
    cleaned = [result[0] for result in normalized]
    tokens = [preprocessing.tokenize(text) for text in cleaned]
    label_sets = load_label_sets()
    punct = {scheme: [labeling.punctuation_labels(t, scheme) for t in tokens] for scheme in SCHEMES}
    casing = [labeling.casing_labels(t, text) for t, text in zip(tokens, cleaned, strict=True)]
    standard = [labeling.project_extended_to_standard(p[0], label_sets) for p in punct["extended"]]
    return {
        "speaker_inventory": speakers,
        "video_context": preprocessing.video_context(texts, TITLES[0]["title"]),
        "video_context_low_comma": preprocessing.video_context(LOW_COMMA_VIDEO),
        "video_context_low_comma_boundaries": [preprocessing.video_context([text]) for text in LOW_COMMA_BOUNDARIES],
        "video_context_tts_title_variant": preprocessing.video_context(texts, TTS_TITLE_VARIANT),
        "normalize": normalized,
        "tokenize": tokens,
        "model_tokens": [[preprocessing.model_token(t.text) for t in row] for row in tokens],
        "model_token_surfaces": [preprocessing.model_token(surface) for surface in MODEL_TOKEN_SURFACES],
        "punctuation_extended": punct["extended"],
        "punctuation_native": punct["native"],
        "projected_standard": standard,
        "casing": casing,
        "joint": [labeling.join_labels(s, c[0]) for s, c in zip(standard, casing, strict=True)],
        "cross_check_standard": [
            labeling.cross_check_standard(t, s, text) for t, s, text in zip(tokens, standard, cleaned, strict=True)
        ],
    }


def text_pipeline_digests() -> dict[str, str]:
    return {stage: canonical_sha256(value) for stage, value in text_pipeline_stages().items()}


# ---- tests ----


@pytest.fixture(scope="module")
def release_file_digests(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    builds = build_fixture_releases(tmp_path_factory.mktemp("release"))
    return {key: digest for name, out in builds.items() for key, digest in release_digests(name, out).items()}


def test_the_release_writes_exactly_the_files_main_wrote(release_file_digests: dict[str, str]) -> None:
    assert sorted(release_file_digests) == sorted(EXPECTED_RELEASE_FILES)


def test_every_release_file_has_the_bytes_main_wrote(release_file_digests: dict[str, str]) -> None:
    changed = sorted(key for key, digest in EXPECTED_RELEASE_FILES.items() if release_file_digests.get(key) != digest)
    assert changed == []


@pytest.fixture(scope="module")
def stage_digests() -> dict[str, str]:
    return text_pipeline_digests()


@pytest.mark.parametrize("stage", sorted(EXPECTED_TEXT_PIPELINE))
def test_each_text_pipeline_stage_gives_what_main_gave(stage_digests: dict[str, str], stage: str) -> None:
    assert stage_digests[stage] == EXPECTED_TEXT_PIPELINE[stage]


def test_the_pins_cover_every_stage(stage_digests: dict[str, str]) -> None:
    assert sorted(EXPECTED_TEXT_PIPELINE) == sorted(stage_digests)
