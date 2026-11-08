# shellcheck shell=bash
# Where the deploy scripts find the repo, the Python environments and the data. Every deploy/*.sh sources this file.
#
#   MELATI_ROOT  the environments (.venv-*), third-party checkouts (ext/) and .env    default /root/melati
#   MELATI_DATA  data, models, results and caches                                   default $MELATI_ROOT/data
#   MELATI_LOGS  the logs of long runs                                              default $MELATI_ROOT/logs
#   PIP_MIRROR   a PyPI mirror for the setup scripts, such as one inside the box's own network (optional)
#
# and MELATI_REPO, this clone, found from where this file is.
#
# Set them in the environment or in $MELATI_ROOT/.env, which also holds HF_TOKEN and is never committed. Keep
# MELATI_DATA outside the repo, so that copying the code tree to a box never touches it.
MELATI_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MELATI_ROOT=${MELATI_ROOT:-/root/melati}
if [ -f "$MELATI_ROOT/.env" ]; then set -a; . "$MELATI_ROOT/.env"; set +a; fi
MELATI_DATA=${MELATI_DATA:-$MELATI_ROOT/data}
MELATI_LOGS=${MELATI_LOGS:-$MELATI_ROOT/logs}
export MELATI_REPO MELATI_ROOT MELATI_DATA MELATI_LOGS
mkdir -p "$MELATI_LOGS" 2>/dev/null || true
# every model and checkout at the revision melati/pins.py names, so a setup run later gets what the dataset was
# made with
pin() { python3 "$MELATI_REPO/melati/pins.py" "$1"; }
checkout() {                                    # $1 a git URL, $2 the folder to hold it at its pinned commit
    [ -d "$2/.git" ] || git clone -q "$1" "$2"
    git -C "$2" checkout -q "$(pin "$1")" 2>/dev/null \
        || { git -C "$2" fetch -q origin && git -C "$2" checkout -q "$(pin "$1")"; }
}
MIRROR=()
if [ -n "${PIP_MIRROR:-}" ]; then
    MIRROR=(--index-url "$PIP_MIRROR" --index-strategy unsafe-best-match)
    case $PIP_MIRROR in               # uv refuses plain http unless the host is named
        http://*) _h=${PIP_MIRROR#http://}; MIRROR+=(--allow-insecure-host "${_h%%/*}") ;;
    esac
fi
