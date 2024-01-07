#!/bin/bash
# shellcheck disable=2207

# Used by nomadt to generate bash completion for argparse.

_nomadt_completion_namespaces() {
	# shellcheck disable=2016
	nomad namespace list -t '{{range $key, $value := .}}{{.Name}} {{end}}'
}

_nomadt_completion_nomad_cmds() {
	nomad --help | sed -n 's/^  \+\([^ ]*\).*/\1/p'
}

_nomadt_completion() {
	# Depends on bash-completion
	if ! declare -f _init_completion 2>/dev/null >&2; then
		return
	fi
	local cur prev words cword split
	_init_completion -s || return
	local i
	for ((i = 1; i <= cword; i++)); do
		if ((i == COMP_CWORD)); then
			break
		fi
		#echo "AAAAAAAAAAA $i ${words[i]} ${COMP_WORDS[i]}"
		case "${words[i]}" in
		-N | --namespace) ((i += 1)) ;;
		--verbose | --autocomplete-info | --autocomplete-install | --version | -h | --help) ;;
		-N* | --namespace=*) ;;
		*)
			# If command is found, complete the command.
			if hash "nomad-${COMP_WORDS[i]}" 2>/dev/null >&2; then
				COMP_WORDS[i]=nomad-${COMP_WORDS[i]}
				_command_offset "$i"
			else
				# Recreate the command line command from words.
				COMP_LINE="nomad"
				COMP_POINT=${#COMP_LINE}
				for (( ; i <= cword; i++)); do
					cur=${words[i]}
					COMP_LINE+=" $cur"
					if ((i <= COMP_CWORD)); then
						((COMP_POINT += 1 + ${#cur}))
					fi
				done
				#echo "${COMP_LINE:0:$COMP_POINT}$COMP_POINT${COMP_LINE:$((COMP_POINT + 1))}"
				# Run go completion. https://github.com/posener/complete/blob/v1/complete.go#L15
				COMPREPLY=($(COMP_LINE=$COMP_LINE COMP_POINT=$COMP_POINT nomad))
				return
			fi
			return
			;;
		esac
	done
	case "$prev" in
	-N | --namespace)
		COMPREPLY=($(compgen -W "$(_nomadt_completion_namespaces)" -- "$cur"))
		;;
	*)
		case "$cur" in
		-*)
			COMPREPLY=($(compgen -W "-N --namespace --verbose --autocomplete-info --autocomplete-install --version -h --help" -- "$cur"))
			;;
		*)
			COMPREPLY=(
				$(compgen -W "$(_nomadt_completion_nomad_cmds)" -- "$cur")
				$(compgen -c -- "nomad-$cur" | sed 's/^nomad-//')
			)
			return
			;;
		esac
		;;
	esac
}

complete -F _nomadt_completion nomadt
