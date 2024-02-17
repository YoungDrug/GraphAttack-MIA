# shellcheck disable=SC2006
py_path=`which python`

run() {
    number=$1
    shift
    for i in $(seq $number); do
      # shellcheck disable=SC2068
      