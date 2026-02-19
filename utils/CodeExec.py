import traceback

class CodeExecutor:

    def __init__(self):
        pass

    def execute(self, code: str, test_cases: list):
        results = []
        execution_env = {}

        try:
            # Compile & execute generated code
            exec(code, execution_env)

            if "solve" not in execution_env:
                return {
                    "status": "error",
                    "error": "No solve() function found in generated code."
                }

            solve_fn = execution_env["solve"]

            for idx, case in enumerate(test_cases):
                try:
                    output = solve_fn(case["input"])
                    passed = output == case["expected_output"]

                    results.append({
                        "test_case_index": idx,
                        "passed": passed,
                        "expected": case["expected_output"],
                        "actual": output
                    })

                except Exception as e:
                    results.append({
                        "test_case_index": idx,
                        "passed": False,
                        "error": str(e)
                    })

            all_passed = all(r["passed"] for r in results if "passed" in r)

            return {
                "status": "success",
                "all_passed": all_passed,
                "results": results
            }

        except Exception as e:
            return {
                "status": "compile_error",
                "error": traceback.format_exc()
            }
