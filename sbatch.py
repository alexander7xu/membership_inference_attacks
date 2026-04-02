from subprocess import check_output


TEMPLATE = r"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J {job_name}
eval "$(conda shell.bash hook)"
conda activate training

command="python ./cli/attack_resnet_cifar10.py {attacker}"

echo $command
$command
echo done
"""


attacker = "lira_online"
job_name = f"{attacker.replace('_', '-')}_resnet_cifar10"
with open("./tmp/_sbatch.bash", "w", encoding="utf-8") as file:
    file.write(TEMPLATE.format(job_name=job_name, attacker=attacker))
out = check_output(["sbatch", f"--output=./logs/{job_name}.log", "./tmp/_sbatch.bash"])
print(attacker, out.decode().strip())
