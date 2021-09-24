import hashlib
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        sha_hash = hashlib.sha256(f.read()).hexdigest()

    final_filename = f'logs/{args.output}-{sha_hash[:8]}.pth'
    shutil.copy(args.input, final_filename)
    print(f'Saved: {final_filename}')
