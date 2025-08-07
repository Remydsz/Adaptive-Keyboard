#!/usr/bin/env python3
"""
Test script to debug the dynamic learning engine core issue
"""

from dynamic_learning_engine import DynamicLearningEngine

def test_hamburger_learning():
    """Test if the core learning algorithm can learn 'hamburger'"""
    print("🧪 Testing Dynamic Learning Engine with hamburger...")
    
    # Create a fresh learning engine
    learning_engine = DynamicLearningEngine()
    
    # Reset to ensure clean test
    learning_engine.reset_learning()
    
    # Simulate basic vocabulary (like our optimized model)
    basic_vocab = {'the', 'and', 'to', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
    
    # Test text with repeated "hamburger"
    test_texts = [
        "I like hamburger hamburger hamburger",
        "The hamburger was good",
        "hamburger is my favorite food",
        "We should get hamburger for lunch",
        "hamburger hamburger hamburger hamburger"
    ]
    
    print(f"📊 Initial state: {learning_engine.get_learning_stats()}")
    
    # Process each text
    for i, text in enumerate(test_texts):
        print(f"\n🔍 Processing text {i+1}: '{text}'")
        try:
            stats = learning_engine.learn_from_text(text, basic_vocab)
            print(f"   Unknown words found: {stats.get('unknown_words_found', 0)}")
            print(f"   New words learned: {stats.get('new_words_learned', 0)}")
            print(f"   Words promoted: {stats.get('words_promoted', [])}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📈 Final stats: {learning_engine.get_learning_stats()}")
    
    # Test predictions
    print(f"\n🔮 Testing predictions for 'hamb':")
    predictions = learning_engine.get_learned_words_for_completion("hamb")
    print(f"   Predictions: {predictions}")

def test_osteoporosis_learning():
    """Test if the core learning algorithm can learn 'osteoporosis'"""
    print("\n\n🧪 Testing Dynamic Learning Engine with osteoporosis...")
    
    # Create a fresh learning engine  
    learning_engine = DynamicLearningEngine()
    learning_engine.reset_learning()
    
    basic_vocab = {'the', 'and', 'to', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
    
    # Simulate user's actual input with 16+ osteoporosis
    test_text = "osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis osteoporosis"
    
    print(f"📊 Initial state: {learning_engine.get_learning_stats()}")
    print(f"🔍 Processing: '{test_text[:50]}...'")
    
    try:
        stats = learning_engine.learn_from_text(test_text, basic_vocab)
        print(f"   Unknown words found: {stats.get('unknown_words_found', 0)}")
        print(f"   New words learned: {stats.get('new_words_learned', 0)}")
        print(f"   Words promoted: {stats.get('words_promoted', [])}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📈 Final stats: {learning_engine.get_learning_stats()}")

if __name__ == "__main__":
    test_hamburger_learning()
    test_osteoporosis_learning()
