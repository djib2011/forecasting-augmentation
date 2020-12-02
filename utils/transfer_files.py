import os
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true', help='Don\'t copy anything, just print the commands.')
parser.add_argument('--host', default='fsu=gpu', help='IP or alias of the host you want files transferred to.')
parser.add_argument('--dir', default='results', help='Directory you want transferred.')

args = parser.parse_args()

remote_dir = 'projects/aug-experiment/' + args.dir
local_dir = args.dir

os.system('ssh {} "ls {}" > /tmp/server_results.log'.format(args.host, remote_dir))

with open('/tmp/server_results.log') as f:
    remote = [line.rstrip('\n') for line in f]

local = [x.name for x in Path(local_dir).glob('*')]

to_be_uploaded = [str(Path(local_dir) / f) for f in set(local).difference(remote)]

print('{:>4} log files found locally.'.format(len(local)))
print('{:>4} log files found in remote server.'.format(len(remote)))
print('{:>4} local log files are not in remote.'.format(len(to_be_uploaded)))

for f in tqdm(to_be_uploaded):
    if args.debug:
        print('scp -r {} {}:{}'.format(f, args.host, remote_dir))
    else:
        os.system('scp -r {} {}:{}'.format(f, args.host, remote_dir))