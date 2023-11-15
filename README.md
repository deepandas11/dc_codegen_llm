# dc_codegen_llm

## Input Schema

```JSON
{
  "inputs": [
    {
      "data": [
        "def factorial(int n):"
      ],
      "name": "prompt",
      "shape": [
        1
      ],
      "datatype": "BYTES"
    }
  ]
}
```

## Output Schema

```JSON
{
  "outputs": [
    {
      "data": [
        "data"
      ],
      "name": "generated_result",
      "shape": [
        1
      ],
      "datatype": "BYTES"
    }
  ]
}
```
