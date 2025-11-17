import argparse
import math
import parse
parser = parse.Parser('''{:g}
{:d}
{:d} {:g}
''')


def parse(filename):
    with open(filename) as file:
        data = file.read()
    return parser.parse(data)


def validate(fa, fb):
    a = parse(fa)
    if a is None:
        return False, f'Failed to parse {fa!r}'
    b = parse(fb)
    if b is None:
        return False, f'Failed to parse {fb!r}'
    bad = []
    if not math.isclose(a[0], b[0], rel_tol=1e-8):
        bad.append('min-dist')
    if not abs(a[1] - b[1]) <= 1:
        bad.append('hit-time-step')
    if a[2] != b[2]:
        bad.append('gravity-device-id')
    if not math.isclose(a[3], b[3], abs_tol=6e4):
        bad.append('missile-cost')
    if bad:
        return False, ','.join(bad)
    return True, 'ok'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fa')
    parser.add_argument('fb')
    ok, msg = validate(**vars(parser.parse_args()))
    print(msg)
    raise SystemExit(not ok)


if __name__ == '__main__':
    main()


