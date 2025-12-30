"""Test ShareGPT dataset loader."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizers import Tokenizer
from src.eztinker.dataset.sharegpt import ShareGPTDataset


def test_dialect_a_from_value():
    """Test loading dialect A (from/value) format."""
    print("\n=== Test 1: Dialect A (from/value) ===")

    # Load JSON file
    dataset = ShareGPTDataset(
        file_path="examples/sharegpt_dialect_a.json",
        tokenizer=None,  # We'll test without tokenizer first
        strict=True,
        max_samples=5
    )

    print(f"Loaded {len(dataset)} conversations")

    # Verify first conversation structure
    conv = dataset[0]
    print(f"\nConversation 0:")
    print(f"  ID: {conv['id']}")
    print(f"  System: {conv['system'][:50]}...")
    print(f"  Turns: {len(conv['turns'])}")
    print(f"  Dataset: {conv['dataset']}")

    # Show normalized turns
    for i, (role, content) in enumerate(conv['turns']):
        print(f"  Turn {i}: {role}")
        print(f"    Content: {content[:60]}...")

    assert len(conv['turns']) == 4
    assert conv['turns'][0] == ("user", "What is the capital of France?")
    assert conv['turns'][1] == ("assistant", "The capital of France is Paris.")
    print("✅ Dialect A validation passed")
    return True


def test_dialect_b_role_content():
    """Test loading dialect B (role/content) format."""
    print("\n=== Test 2: Dialect B (role/content) ===")

    # Load JSON file
    dataset = ShareGPTDataset(
        file_path="examples/sharegpt_dialect_b.json",
        tokenizer=None,
        strict=True,
        max_samples=5
    )

    print(f"Loaded {len(dataset)} conversations")

    # Verify first conversation structure
    conv = dataset[0]
    print(f"\nConversation 0:")
    print(f"  ID: {conv['id']}")
    print(f"  System: {conv['system'][:50]}...")
    print(f"  Turns: {len(conv['turns'])}")
    print(f"  Dataset: {conv['dataset']}")

    # Show normalized turns
    for i, (role, content) in enumerate(conv['turns']):
        print(f"  Turn {i}: {role}")
        print(f"    Content: {content[:60]}...")

    assert len(conv['turns']) == 2
    assert conv['turns'][0][0] == "user"
    assert conv['turns'][1][0] == "assistant"
    print("✅ Dialect B validation passed")
    return True


def test_jsonl_format():
    """Test JSONL file format."""
    print("\n=== Test 3: JSONL Format ===")

    # Test JSONL dialect A
    dataset_a = ShareGPTDataset(
        file_path="examples/sharegpt_dialect_a.jsonl",
        tokenizer=None,
        strict=True
    )

    print(f"JSONL Dialect A loaded: {len(dataset_a)} conversations")

    # Test JSONL dialect B
    dataset_b = ShareGPTDataset(
        file_path="examples/sharegpt_dialect_b.jsonl",
        tokenizer=None,
        strict=True
    )

    print(f"JSONL Dialect B loaded: {len(dataset_b)} conversations")

    assert len(dataset_a) == 1
    assert len(dataset_b) == 1
    print("✅ JSONL format validation passed")
    return True


def test_format_conversation_qwen2():
    """Test Qwen2 template formatting."""
    print("\n=== Test 4: Qwen2 Template Formatting ===")

    dataset = ShareGPTDataset(
        file_path="examples/sharegpt_dialect_b.json",
        tokenizer=None,
        strict=True,
        max_samples=1
    )

    conv = dataset[0]
    formatted = dataset.format_conversation_qwen2(conv)

    print("\nFormatted conversation:")
    print(formatted[:200] + "...")
    print("...")

    # Check template structure
    assert "<|im_start|>system" in formatted
    assert "<|im_start|>user" in formatted
    assert "<|im_start|>assistant" in formatted
    assert "<|im_end|>" in formatted
    print("✅ Qwen2 formatting validation passed")
    return True


def test_validation_and_error_handling():
    """Test validation and error handling."""
    print("\n=== Test 5: Validation and Error Handling ===")

    # Create temporary invalid file
    temp_file = "examples/test_invalid.json"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump([{
            "id": "invalid_001",
            "messages": [
                {"role": "assistant", "content": "This starts with assistant!"}
            ]
        }], f)

    # Test with strict=False (should warn but not crash)
    try:
        dataset = ShareGPTDataset(
            file_path=temp_file,
            tokenizer=None,
            strict=False  # Non-strict mode
        )
        print(f"Non-strict mode: {len(dataset)} valid conversations")
        print(f"Dropped: {dataset.stats['invalid_conversations']} invalid conversations")
        assert len(dataset) == 0  # No valid conversations
        print("✅ Non-strict mode validation passed")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        # Cleanup
        Path(temp_file).unlink(missing_ok=True)

    return True


def test_export_to_jsonl():
    """Test export to JSONL format."""
    print("\n=== Test 6: Export to JSONL ===")

    # Load dataset
    dataset = ShareGPTDataset(
        file_path="examples/sharegpt_dialect_a.json",
        tokenizer=None,
        strict=True,
        max_samples=2
    )

    # Export to JSONL
    output_path = "examples/test_export.jsonl"
    dataset.export_to_jsonl(output_path)

    # Load back and verify
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Exported {len(lines)} conversations to JSONL")

        # Parse first line
        exported = json.loads(lines[0])
        print(f"First exported conversation:")
        print(f"  ID: {exported['id']}")
        print(f"  System: {exported['system'][:30]}...")
        print(f"  Messages: {len(exported['messages'])}")

        # Verify format (should be role/content)
        assert "messages" in exported
        assert exported["messages"][0]["role"] == "user"
        assert exported["messages"][0]["content"] is not None

    # Cleanup
    Path(output_path).unlink(missing_ok=True)
    print("✅ Export to JSONL validation passed")
    return True


def test_multiple_messages():
    """Test multi-turn conversations."""
    print("\n=== Test 7: Multi-turn Conversations ===")

    # Load vicuna-style conversation (has multiple turns)
    dataset = ShareGPTDataset(
        file_path="examples/sharegpt_dialect_b.json",
        tokenizer=None,
        strict=True
    )

    # Find the multi-turn conversation (vicuna_003)
    multi_turn = None
    for conv in dataset.conversations:
        if len(conv['turns']) > 2:
            multi_turn = conv
            break

    if multi_turn:
        print(f"\nFound multi-turn conversation:")
        print(f"  ID: {multi_turn['id']}")
        print(f"  Turns: {len(multi_turn['turns'])}")
        for i, (role, content) in enumerate(multi_turn['turns']):
            print(f"  Turn {i} ({role}): {content[:50]}...")

        # Verify alternation
        for i in range(len(multi_turn['turns']) - 1):
            assert multi_turn['turns'][i][0] != multi_turn['turns'][i+1][0], \
                f"Turn {i} and {i+1} have same role"

        print("✅ Multi-turn validation passed")
        return True
    else:
        print("⚠️  No multi-turn conversation found")
        return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running ShareGPT Dataset Loader Tests")
    print("=" * 60)

    tests = [
        test_dialect_a_from_value,
        test_dialect_b_role_content,
        test_jsonl_format,
        test_format_conversation_qwen2,
        test_validation_and_error_handling,
        test_export_to_jsonl,
        test_multiple_messages
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    exit(run_all_tests())