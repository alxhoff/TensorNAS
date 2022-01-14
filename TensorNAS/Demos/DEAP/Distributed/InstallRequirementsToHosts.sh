#!/bin/bash

host_file="hosts"

if ! command -v pdsh &> /dev/null
then
    echo "pdsh could not be found, please install and rerun script"
    exit 1
fi

while getopts u:h: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        h) host_file=${OPTARG};;
    esac
done

echo "Copying requirements.txt to all hosts"

while IFS= read -r line; do
    ip=${line%% *}
    scp requirements.txt $username@$ip:~/
done < "hosts"

echo "Installing requirements on all hosts"

pdsh -w ^hosts -l $username -R ssh "python3 -m pip install -r ~/requirements.txt"