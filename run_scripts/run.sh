#!/usr/bin/env bash

# --------- code to ensure docker is running properly ------------

# Allow script to exit
trap "echo; exit" INT

OPT_B='false'
OPT_S='false'
DRY_RUN='false'

path=''
gpu=''
while getopts ':bsdg:c:m:p:' 'OPTKEY'; do
    case ${OPTKEY} in
        'p')
            path=${OPTARG}
            ;;
        'g')
            gpu=${OPTARG}
            re_isanum='^[0-9]+$'                # Regex: match whole numbers only
            if ! [[ $gpu =~ $re_isanum ]] ; then
                echo "Error: GPU id must be a positive integer."
                exit_abnormal
            fi
            ;;
        'c')
            cpus=${OPTARG}
            ;;
        'm')
            memory=${OPTARG}
            ;;

        'b')
            OPT_B='true'
            ;;
        's')
            OPT_S='true'
            ;;
        'd')
            DRY_RUN='true'
            ;;
        '?')
            echo "INVALID OPTION -- ${OPTARG}" >&2
            exit 1
            ;;
        ':')
            echo "MISSING ARGUMENT for option -- ${OPTARG}" >&2
            exit 1
            ;;
        *)
            echo "UNIMPLEMENTED OPTION -- ${OPTKEY}" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${path}" ]]; then
    echo "-p path to subtasks must be provided."
    exit 1
fi

# GPU not provided
if [[ -z "${gpu}" ]]; then
    CONTAINER_NAME=malware-certification-cpu
    gpu_args=""
else
    CONTAINER_NAME=malware-certification-gpu-${gpu}
    gpu_args="--gpu ${gpu}"
fi

if [[ $DRY_RUN == 'true' ]]; then
    echo "[DRY_RUN] enabled";
fi

# Start docker if not running
docker container inspect $CONTAINER_NAME > /dev/null
status=$?
[ $status -eq 0 ] && echo "Container already running, -g -c -m will be ignored" || {
    if [[ -z "${cpus}" ]] || [[ -z "${memory}" ]]; then
        echo "-c, -m must all be provided to start container"
        exit 1
    fi
    cmd="python3 ../docker/deploy.py ${gpu_args} --cpus ${cpus} --memory ${memory} --non-interactive"
    echo "Command to run: $cmd"
    if [[ $DRY_RUN == 'false' ]]; then
        echo "Starting container" ;
        eval $(ssh-agent);
        ssh-add;
        eval $cmd;
    else
        echo "[DRY_RUN] Don't start container"
    fi
}


# Fetch the list of jobs to run from other scripts and make the run functiion
run() {
    echo "Sourcing the jobs from $path"
    source $path

    echo "---------------------------"
    echo "The list of jobs to run are:"
    for job in "${jobs[@]}"; do
        echo -e "\t$job"
    done

    for job in "${jobs[@]}"; do
        if [[ $DRY_RUN = "false" ]]; then
            echo "Running: docker exec $CONTAINER_NAME $job"
            docker exec --tty $CONTAINER_NAME $job
        fi
    done
}

if [[ ${OPT_B} = "true" ]]; then
    fname=$(basename -- $path)
    fname=${fname/\.sh/}
    fname=job-outputs/$fname
    if [[ -e $fname.out || -L $fname.out ]] ; then
        i=0
        while [[ -e $fname-$i.out || -L $fname-$i.out ]] ; do
            let i++
        done
        fname=$fname-$i
    fi
    stdout=$fname.out
    stderr=$fname.err
    echo "save stdout to: $stdout";
    echo "save stderr to: $stderr";
fi

# Executes the task
if [[ ${OPT_B} = "false" || ${DRY_RUN} = "true" ]]; then
    run
else
    if [[ $DRY_RUN == 'false' ]]; then
        run 1> $stdout 2> $stderr &
    fi
fi

# Clean up container if needed
if [ "${OPT_S}" = "true" ]; then
    echo "Stopping the container"
    docker stop $CONTAINER_NAME
fi
