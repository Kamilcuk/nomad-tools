#!/bin/bash
set -xeuo pipefail

# test all cases from:
# https://docs.docker.com/engine/reference/commandline/container_cp/

# Small wrapper to execute nomad-cp
nomad-cp() { python -m nomad_tools.nomad_cp -vv "$@"; }

# Handle that this command will fail
willfail() { if "$@"; then exit 123; fi; }

# Write to a file. Function exists so it is visible in set -x.
write() { echo "$1" >"$2"; }

# Show the error location.
trap_err() { echo "Error on line $1"; }
trap 'trap_err $LINENO' ERR

# We require a temporary directory with source and destination.
tmpd=$(mktemp -d)
trap_exit() {
	echo "EXIT"
	cd /
	rm -vrf "$tmpd"
}
trap 'trap_exit' EXIT
cd "$tmpd"

# Populate source directory and make destination directory.
mkdir -v src src/d dst
write 'text1' src/1
write 'text2' src/d/2

{
	echo 'SRC_PATH specifies a file'
	{
		echo 'DEST_PATH does not exist'
		echo 'the file is saved to a file created at DEST_PATH'
		[[ ! -e dst/1 ]]
		nomad-cp src/1 dst/1
		[[ -e dst/1 ]]
		[[ $(<src/1) = $(<dst/1) ]]
	}
	{
		echo 'DEST_PATH does not exist and ends with /'
		echo 'Error condition: the destination directory must exist.'
		willfail nomad-cp src/1 src/2/
	}
	{
		echo 'DEST_PATH exists and is a file'
		echo "the destination is overwritten with the source file's contents"
		[[ -e dst/1 ]]
		nomad-cp src/d/2 dst/1
		[[ -e dst/1 ]]
		[[ $(<dst/1) = $(<src/d/2) ]]
	}
	{
		echo 'DEST_PATH exists and is a directory'
		echo 'the file is copied into this directory using the basename from SRC_PATH'
		rm -v dst/1
		nomad-cp src/1 dst
		[[ -e dst/1 ]]
		[[ $(<dst/1) = $(<src/1) ]]
	}
	rm -vrf dst/*
}
{
	echo 'SRC_PATH specifies a directory'
	{
		echo 'DEST_PATH does not exist'
		echo 'DEST_PATH is created as a directory and the contents of the source directory are copied into this directory'
		[[ ! -e dst/d ]]
		nomad-cp src dst/src
		[[ $(<src/1) = $(<dst/src/1) ]]
		[[ $(<src/d/2) = $(<dst/src/d/2) ]]
		rm -vrf dst/src
	}
	{
		echo 'DEST_PATH exists and is a file'
		echo 'Error condition: cannot copy a directory to a file'
		touch dst/1
		willfail nomad-cp src dst/1
		rm -v dst/1
	}
	{
		echo 'DEST_PATH exists and is a directory'
		{
			echo 'SRC_PATH does not end with /. (that is: slash followed by dot)'
			echo 'the source directory is copied into this directory'
			nomad-cp src dst
			[[ $(<src/1) = $(<dst/src/1) ]]
			[[ $(<src/d/2) = $(<dst/src/d/2) ]]
			rm -vrf dst/src
		}
		{
			echo 'SRC_PATH does end with /. (that is: slash followed by dot)'
			echo 'the content of the source directory is copied into this directory'
			nomad-cp src/. dst
			[[ $(<src/1) = $(<dst/1) ]]
			[[ $(<src/d/2) = $(<dst/d/2) ]]
		}
	}
	rm -vrf dst/*
}
