#!/usr/bin/env bash
# Wrapper for the official Colab CLI in containers without an OS keyring.
# Set COLAB_VENV to the venv holding google-colab-cli (default: alongside this file).
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring
exec "${COLAB_VENV:-$(dirname "$0")/colabcli}/bin/colab" "$@"
