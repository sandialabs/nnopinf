#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
failed=0

sorted_dirs=$(
    for dim_dir in "${script_dir}"/dim*-lpopinf/; do
        [[ -d "${dim_dir}" ]] || continue
        dimension="${dim_dir##*/dim}"
        dimension="${dimension%-lpopinf/}"
        printf '%s\t%s\n' "${dimension}" "${dim_dir}"
    done | sort -n | cut -f2-
)

while IFS= read -r dim_dir; do
    [[ -n "${dim_dir}" ]] || continue

    printf 'Running LPOpInf model in %s\n' "${dim_dir}"
    if (
        cd -- "${dim_dir}"
        julia --threads 1 \
            --project=@/Users/ejparis/codes/Norma.jl \
            /Users/ejparis/codes/Norma.jl/src/Norma.jl \
            torsion-in.yaml
    ); then
        :
    else
        printf 'Run failed in %s\n' "${dim_dir}" >&2
        failed=1
    fi
done <<< "${sorted_dirs}"

exit "${failed}"
