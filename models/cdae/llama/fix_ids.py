import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, required=True)
    parser.add_argument('--targe', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    generated_jsons = []
    with open(args.generated, 'r') as f:
        for line in f:
            generated_jsons.append(json.loads(line))

    target_jsons = []
    with open(args.target, 'r') as f:
        for line in f:
            target_jsons.append(json.loads(line))

    for i in range(len(generated_jsons)):
        if generated_jsons[i]['instance_id'] != target_jsons[i]['instance_id']:
            generated_jsons[i]['instance_id'] = target_jsons[i]['instance_id']

    with open(args.output, 'w') as f:
        for line in generated_jsons:
            f.write(json.dumps(line) + '\n')