from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.dashboard_runtime import DashboardRuntime
from wm_shared.config import load_yaml_config
from wm_shared.run_manifest import load_latest_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--family-dir', default=None)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-open-browser', action='store_true')
    args = parser.parse_args()

    family_dir = None
    if args.config:
        cfg = load_yaml_config(args.config)
        family_dir = Path(args.family_dir or cfg['logging']['dir']).resolve()
    elif args.family_dir:
        family_dir = Path(args.family_dir).resolve()
    else:
        raise SystemExit('pass --config or --family-dir')

    manifest = load_latest_manifest(family_dir)
    if manifest is None:
        print(f'[dashboard] no latest run manifest found under {family_dir}; starting idle dashboard shell')
    else:
        print(
            f'[dashboard] found latest run {manifest.identity.run_id} under {family_dir}; '
            'starting inspect-only dashboard shell (no live job reattachment)'
        )

    runtime = DashboardRuntime(
        family_root=family_dir,
        manifest=manifest,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open_browser,
    )
    runtime.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        runtime.close()


if __name__ == '__main__':
    main()
