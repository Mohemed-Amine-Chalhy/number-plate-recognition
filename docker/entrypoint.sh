#!/bin/sh
set -eu

# Fail before Streamlit becomes reachable when a required model artifact is
# missing, corrupt, or inconsistent with its declared contract.
python /opt/app/scripts/doctor.py --models-only

exec "$@"
