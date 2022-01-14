#!/bin/bash
host_file="hosts"

if ! command -v sshpass &> /dev/null
then
    echo "sshpass could not be found, please install and rerun script"
    exit 1
fi

while getopts u:k:h: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        k) pub_key=${OPTARG};;
        h) host_file=${OPTARG};;
    esac
done

if [ -z "$username" ]
then
    echo "No username provided, please provide one with '-u'"
    exit 1
fi

if [ -z "$pub_key" ]
then
    pub_key=$(find ~/.ssh -name "*.pub")
fi

if [ -z "$pub_key" ]
then
    echo "No public key found, create one in ~/.ssh and rerun script"
    exit 1
else
    echo "Found public key:"
    echo $pub_key
fi

IFS= read -s -p "Password: " password; echo

while IFS= read -r line; do
    ip=${line%% *}
    echo "sshpass -p ***** ssh-copy-id -i $pub_key $username@$ip"
    sshpass -p $password ssh-copy-id -f -i $pub_key -o StrictHostKeyChecking=no $username@$ip
done < "hosts"
