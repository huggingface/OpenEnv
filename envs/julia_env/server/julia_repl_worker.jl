#!/usr/bin/env julia

"""
Julia REPL Worker for Process Pool

This script runs as a persistent Julia process that accepts code via stdin,
executes it, and returns results via stdout with delimiters.

Protocol:
- Input: Code block followed by "<<<END_CODE>>>"
- Output: Results with status markers:
  - "<<<START_OUTPUT>>>" - stdout begins
  - "<<<START_ERROR>>>" - stderr begins
  - "<<<EXIT_CODE:N>>>" - exit code (0 = success, 1 = error)
  - "<<<END_EXECUTION>>>" - execution complete
"""

# Preload commonly used modules to reduce JIT compilation overhead
# This is done once at worker startup, so subsequent code executions are faster
using Test

# Warm up the Test framework and common operations by running simple tests
# This pre-compiles the @test and @testset macros plus common Julia operations
let
    # Warm up Test framework
    @testset "warmup" begin
        @test 1 + 1 == 2
        @test "hello" == "hello"
        @test [1, 2, 3] == [1, 2, 3]
    end

    # Warm up common string operations
    occursin("a", "abc")
    replace("hello", "l" => "x")
    split("a,b,c", ",")
    join(["a", "b"], ",")
    lowercase("ABC")
    uppercase("abc")
    strip("  hello  ")

    # Warm up common array operations
    arr = [3, 1, 2]
    push!(arr, 4)
    pop!(arr)
    append!(arr, [5, 6])
    sort(arr)
    reverse(arr)
    length(arr)
    isempty(Int[])

    # Warm up common math operations
    div(10, 3)
    mod(10, 3)
    abs(-5)
    sqrt(4.0)
    floor(Int, 3.7)
    ceil(Int, 3.2)

    # Warm up Dict operations
    d = Dict("a" => 1, "b" => 2)
    get(d, "a", 0)
    haskey(d, "a")
    keys(d)
    values(d)
end

# Delimiters for communication protocol
const START_OUTPUT = "<<<START_OUTPUT>>>"
const START_ERROR = "<<<START_ERROR>>>"
const EXIT_CODE_PREFIX = "<<<EXIT_CODE:"
const END_EXECUTION = "<<<END_EXECUTION>>>"
const END_CODE = "<<<END_CODE>>>"

"""
Execute code and capture output using pipes
"""
function execute_code(code::String)
    # Initialize return values
    stdout_str = ""
    stderr_str = ""
    exit_code = 0

    # Create pipes for output capture
    out_pipe = Pipe()
    err_pipe = Pipe()

    try
        # Execute with output redirected to pipes
        redirect_stdout(out_pipe) do
            redirect_stderr(err_pipe) do
                try
                    # Execute the code using include_string which properly handles
                    # multiple statements including 'using' statements
                    include_string(Main, code)
                catch e
                    # Execution error - write to stderr
                    exit_code = 1
                    showerror(stderr, e, catch_backtrace())
                    println(stderr)
                end
            end
        end

        # Close write ends to signal EOF to readers
        Base.close(out_pipe.in)
        Base.close(err_pipe.in)

        # Read captured output
        stdout_str = read(out_pipe.out, String)
        stderr_str = read(err_pipe.out, String)

        # Close read ends
        Base.close(out_pipe.out)
        Base.close(err_pipe.out)

    catch e
        # Worker error
        exit_code = 1

        # Try to close pipes
        try
            Base.close(out_pipe)
            Base.close(err_pipe)
        catch
        end

        stderr_str = "Worker error: " * sprint(showerror, e)
    end

    return (stdout_str, stderr_str, exit_code)
end

"""
Main loop: read code, execute, return results
"""
function main()
    # Signal that worker is ready
    println(stderr, "Julia worker ready")
    flush(stderr)

    while true
        try
            # Read code until END_CODE delimiter
            code_lines = String[]

            while true
                if eof(stdin)
                    println(stderr, "Worker received EOF, shutting down")
                    return
                end

                line = readline(stdin)

                # Check for end of code
                if line == END_CODE
                    break
                end

                push!(code_lines, line)
            end

            # If no code received, continue
            if isempty(code_lines)
                # Send empty response
                println(START_OUTPUT)
                println(START_ERROR)
                println(EXIT_CODE_PREFIX, 0, ">>>")
                println(END_EXECUTION)
                flush(stdout)
                continue
            end

            code = join(code_lines, "\n")

            # Execute code and capture output
            (stdout_str, stderr_str, exit_code) = execute_code(code)

            # Send results with delimiters
            println(START_OUTPUT)
            print(stdout_str)
            flush(stdout)

            println(START_ERROR)
            print(stderr_str)
            flush(stdout)

            println(EXIT_CODE_PREFIX, exit_code, ">>>")
            println(END_EXECUTION)
            flush(stdout)

        catch e
            # Worker error - report and continue
            println(stderr, "Worker error: ", e)
            flush(stderr)

            # Send error response
            println(START_OUTPUT)
            println(START_ERROR)
            println("Worker internal error: ", e)
            println(EXIT_CODE_PREFIX, 1, ">>>")
            println(END_EXECUTION)
            flush(stdout)
        end
    end
end

# Run main loop
main()

