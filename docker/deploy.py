#!/usr/bin/env python3

import argparse
import os
import subprocess
import re

# Move working dir to the correct one
file_path = os.path.realpath(__file__)
os.chdir(os.path.dirname(file_path))

TAG_NAME = "malware-certification/malware-certification"
CONTAINER_NAME = "malware-certification"

USER = os.getlogin()
UID = os.getuid()
GID = os.getgid()
if USER == "root":
    USER = "app"
    UID += 1000
    GID += 1000

def main(args):
    global BASE_IMAGE, TAG_NAME, CONTAINER_NAME

    # build the Docker image
    tag_name = TAG_NAME
    container_name = CONTAINER_NAME
    build_args = [
        "--build-arg", f"UID={UID}",
        "--build-arg", f"GID={GID}",
        "--build-arg", f"USER={USER}",
    ]
    env = {
        **os.environ,
        "DOCKER_BUILDKIT": str(1),
    }
    subprocess.check_call(
        ["docker", "build", "--force-rm", "--ssh", "default", "--tag", tag_name] 
        + build_args
        + [os.getcwd()], env=env)
    src_dir = os.path.dirname(os.getcwd())
    volume_mappings = []
    # Mount src
    exe_dirs = ["src"]
    data_dirs = ["data"]
    output_dirs = ["configs", "outputs"]
    dirs = exe_dirs + data_dirs + output_dirs
    for dir in dirs:
        os.makedirs(f"{src_dir}/{dir}", exist_ok=True)
        volume_mappings += ['-v', f"{src_dir}/{dir}:/app/{dir}:z"]

    rsrc_args = []
    if args.gpus != "none":
        container_name += "-gpu-" + re.sub(r",\s*", "_", args.gpus)
        rsrc_args.extend(["--ipc", "host", "--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=" + args.gpus])
    else:
        container_name += "-cpu"

    if args.cpus is not None:
        rsrc_args.append('--cpus={}'.format(args.cpus))
    if args.memory is not None:
        rsrc_args.append("--memory={}".format(args.memory))

    if args.non_interactive:
        interactive_option = '-dt'
    else:
        interactive_option = "-it"

    if args.name is not None:
        container_name = args.name

    user_args = ["-u", USER, "--userns=keep-id"]

    commands = (
        ["docker", "run", "--rm", interactive_option, "--name", container_name, "-P"]
        + volume_mappings
        + user_args
        + rsrc_args
        + ["{}:latest".format(tag_name), "/bin/bash"]
    )
    print(" ".join(commands))
    subprocess.call(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy script")
    parser.add_argument(
        "--non-interactive",
        help="Specify this flag to diable interactive shell (for running scripts)",
        action="store_true",
    )
    parser.add_argument(
        "--gpus",
        help="GPUs accessible inside the container. Possible values include: 'all', 'none', a comma-separated list of "
             "GPU UUIDs or index(es)",
        type=str,
        default="none",
    )
    parser.add_argument(
        "--memory", "-m",
        help="Maximum amount of memory the container can use, e.g. '2g'",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cpus",
        help="Available resources the container can use, e.g. '2' allows the container to use at most 2 CPUs",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name",
        help="Name to assign to container",
        default=None,
    )
    args = parser.parse_args()
    main(args)
