#!/bin/sh
set -eu

# Fail before Streamlit becomes reachable when required model artifacts are
# missing, corrupt, or disallowed by the selected NPR_ENVIRONMENT policy.
python /opt/app/scripts/doctor.py --models-only

exec "$@"
