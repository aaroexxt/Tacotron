#!/bin/bash

abort()
{
    echo >&2 '
***************
*** ABORTED ***
***************
'
    echo "An error occurred :( Exiting..." >&2
    exit 1;
}

#set -e -u;

if [[ $(id -u) -ne 0 ]]
  then echo "Sorry, but it appears that you didn't run this script as root. Please run it as a root user!";
  exit 1;
fi

echo "Skicka init script"; echo "---------------------------------------------------";
sudo apt-get install -y golang-go g++ git mercurial;
echo "Installed required packages.";
echo "GOPATH=$HOME/.go" >> ~/.bashrc
source ~/.bashrc
go get github.com/google/skicka
skicka init
echo "When you press enter, a window will launch. Please enter your email and password for use with skicka.";
skicka ls
echo "Done setting up google drive integration. Pass the --upload-drive flag to train.py for automatic uploading of training."