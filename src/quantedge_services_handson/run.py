"""QuantEdge Hands-On Runner.

Usage:
    python -m quantedge_services_handson.run                  # all weeks 00-12
    python -m quantedge_services_handson.run 00               # single week
    python -m quantedge_services_handson.run 00,01,02         # comma-separated
    python -m quantedge_services_handson.run 00-04            # range
    python -m quantedge_services_handson.run 00-02,05,10-12   # mixed
    python -m quantedge_services_handson.run 03 --force       # with force flag
"""

import argparse
import importlib

ALL_WEEKS = [f"{i:02d}" for i in range(13)]  # 00 through 12


def parse_weeks(spec: str) -> list[str]:
    weeks = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            s, e = int(start), int(end)
            weeks.extend(f"{i:02d}" for i in range(s, e + 1))
        else:
            weeks.append(part.zfill(2))
    return sorted(set(weeks))


def run_week(week: str, force: bool = False) -> dict:
    module_name = f"quantedge_services_handson.week{week}.main"
    try:
        mod = importlib.import_module(module_name)
        return mod.run(force=force)
    except ImportError:
        print(f"  WEEK {week} — no main.py yet, skipping")
        return {"week": week, "status": "no main.py"}
    except TypeError:
        # week01+ main.run() may not accept force=
        mod = importlib.import_module(module_name)
        return mod.run()
    except Exception as e:
        print(f"  WEEK {week} — FAILED: {e}")
        return {"week": week, "status": "error", "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantEdge Hands-On Runner")
    parser.add_argument("weeks", nargs="?", default=None,
                        help="Weeks: 00 | 00,01,02 | 00-04 | 00-02,05,10-12")
    parser.add_argument("--force", action="store_true", help="Force re-generate")
    args = parser.parse_args()

    weeks = parse_weeks(args.weeks) if args.weeks else ALL_WEEKS

    print("=" * 50)
    print(f"  Hands-On Runner — weeks: {', '.join(weeks)}")
    print("=" * 50)

    for week in weeks:
        print()
        run_week(week, force=args.force)

    print("\n" + "=" * 50)
    print("  Done.")
    print("=" * 50)


if __name__ == "__main__":
    main()
