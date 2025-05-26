# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import setuptools

_deps = [
    # "transformers",
    # "torch",
    # "librosa",
    "descript-audio-codec",
    "descript-audiotools @ git+https://github.com/descriptinc/audiotools",  # temporary fix as long as 0.7.4 is not published
    "protobuf"
]

extras_dev_deps = [
    "black~=23.1",
    "isort>=5.5.4",
    "ruff>=0.0.241,<=0.0.259",
]

extras_training_deps = [
    "jiwer",
    "wandb",
    "accelerate",
    "evaluate",
    "datasets[audio]>=2.14.5",
]

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# read version - look for it in root __init__.py or set fallback
try:
    with open(os.path.join(here, "__init__.py"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break
        else:
            raise RuntimeError("Unable to find version string.")
except FileNotFoundError:
    # Fallback version if __init__.py doesn't exist
    version = "0.1.0"

setuptools.setup(
    name="dac_encodec",
    version=version,
    description="a Wrapper to make DAC identical to EnCodec.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=[
        # List your .py files here (without .py extension)
        # For example, if you have modeling_dac.py, add "modeling_dac"
    ],
    install_requires=_deps,
    extras_require={
        "dev": extras_dev_deps,
        "training": extras_training_deps,
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
