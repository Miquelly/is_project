import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nome1', help='Primeiro nome')
parser.add_argument('--nome2', help='Segundo nome')

args = parser.parse_args()

print(f'Primeiro nome: {args.nome1}')
print(f'Segundo nome: {args.nome2}')

