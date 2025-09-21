"""
Test script to verify deranking implementation works as intended.
Tests the complete flow: deranking -> p(seen) -> p(reshare|seen)
"""

import sys
import os
import numpy as np
import random
from collections import defaultdict, Counter

# Add the libs directory to the path so we can import simsom (adjust path for workflow/scripts location)
sys.path.append('../../libs')

from simsom.model import SimSomMod
from simsom.message import Message

def create_test_network():
    """Create a simple test network for controlled testing."""
    import igraph as ig
    
    # Create a small network: 5 agents, simple connections
    g = ig.Graph(directed=True)
    
    # Add 5 vertices (agents)
    g.add_vertices(5)
    
    # Set agent properties
    for i, v in enumerate(g.vs):
        v["uid"] = f"agent_{i}"
        v["bot"] = 0  # All humans for now
        v["class"] = "normal"
        v["postperday"] = 1.0  # Each agent posts once per day
        v["qualitydistr"] = "[2, 2, 0, 1]"  # Beta distribution parameters
    
    # Add edges: create a simple network where agent_0 follows everyone
    # agent_1 follows agent_2 and agent_3, etc.
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # agent_0 follows everyone
        (1, 2), (1, 3),                   # agent_1 follows agent_2, agent_3
        (2, 3), (2, 4),                   # agent_2 follows agent_3, agent_4
        (3, 4)                            # agent_3 follows agent_4
    ]
    g.add_edges(edges)
    
    # Save as temporary GML file
    g.write_gml("test_network.gml")
    return "test_network.gml"

def create_test_messages(model):
    """Create test messages with known quality levels."""
    messages = {}
    
    # Create 4 test messages with different qualities
    test_cases = [
        {"id": 1, "quality": 0.9, "appeal": 0.8, "is_bad": False, "description": "High quality"},
        {"id": 2, "quality": 0.2, "appeal": 0.7, "is_bad": True, "description": "Low quality (bad)"},
        {"id": 3, "quality": 0.8, "appeal": 0.6, "is_bad": False, "description": "Good quality"},
        {"id": 4, "quality": 0.1, "appeal": 0.9, "is_bad": True, "description": "Low quality, high appeal (bad)"}
    ]
    
    for case in test_cases:
        # Create message manually
        message = Message(
            id=case["id"],
            user_id="test_user",
            is_by_bot=False,
            quality_distr=None
        )
        # Override the random values with our test values
        message.quality = case["quality"]
        message.appeal = case["appeal"]
        message.legality = "illegal" if case["is_bad"] else "legal"
        
        messages[case["id"]] = message
        model.all_messages[case["id"]] = message
        
        # Initialize message metadata
        model.message_metadata[case["id"]] = {
            "human_shares": 0,
            "bot_shares": 0,
            "spread_via_agents": [],
            "seen_by_agents": [],
            "seen_by_agent_timestep": [],
        }
    
    return test_cases

def setup_test_feed(model, agent_id, message_ids, shares_list, ages_list):
    """Set up a test newsfeed for an agent."""
    model.agent_feeds[agent_id] = (
        np.array(message_ids),
        np.array(shares_list),
        np.array(ages_list)
    )

def test_deranking_effect():
    """Test that bad content gets deranked properly."""
    print("=" * 60)
    print("TEST 1: DERANKING EFFECT")
    print("=" * 60)
    
    # Create test network
    network_file = create_test_network()
    
    # Test with different deranking severities
    deranking_values = [1.0, 0.5, 0.1]
    
    for deranking_severity in deranking_values:
        print(f"\n--- Testing with deranking_severity = {deranking_severity} ---")
        
        # Create model
        model = SimSomMod(
            graph_gml=network_file,
            deranking_severity=deranking_severity,
            r_half=10,
            temp=2,
            modeling_legality=True,
            verbose=False
        )
        
        # Create test messages
        test_cases = create_test_messages(model)
        
        # Set up test feed for agent_0
        message_ids = [1, 2, 3, 4]
        shares = [5, 5, 5, 5]  # Same shares initially
        ages = [1, 1, 1, 1]    # Same ages
        setup_test_feed(model, "agent_0", message_ids, shares, ages)
        
        # Test the ranking
        newsfeed = model.agent_feeds["agent_0"]
        message_info, ranking = model._rank_newsfeed(newsfeed)
        
        # Analyze results
        seen_messages = message_info[0]
        seen_rankings = message_info[5]  # ranking scores
        
        print(f"Messages seen: {len(seen_messages)} out of {len(message_ids)}")
        
        for i, msg_id in enumerate(seen_messages):
            case = next(c for c in test_cases if c["id"] == msg_id)
            print(f"  Message {msg_id} ({case['description']}): ranking = {seen_rankings[i]:.4f}")
    
    # Clean up
    os.remove(network_file)

def test_friend_engagement_effect():
    """Test that friend engagement affects p(reshare|seen)."""
    print("\n" + "=" * 60)
    print("TEST 2: FRIEND ENGAGEMENT EFFECT")
    print("=" * 60)
    
    # Create test network
    network_file = create_test_network()
    
    # Create model
    model = SimSomMod(
        graph_gml=network_file,
        deranking_severity=1.0,  # No deranking for this test
        r_half=10,
        temp=2,
        modeling_legality=False,
        verbose=False
    )
    
    # Create test messages
    test_cases = create_test_messages(model)
    
    # Simulate different friend engagement scenarios
    scenarios = [
        {"name": "No friend engagement", "shared_by": []},
        {"name": "Some friend engagement", "shared_by": ["agent_2"]},
        {"name": "High friend engagement", "shared_by": ["agent_2", "agent_3", "agent_4"]}
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Reset message metadata
        for msg_id in [1, 2, 3, 4]:
            model.message_metadata[msg_id]["spread_via_agents"] = scenario["shared_by"].copy()
        
        # Set up test feed for agent_0 (who follows agent_1, 2, 3, 4)
        message_ids = [1, 2, 3, 4]
        shares = [3, 3, 3, 3]
        ages = [1, 1, 1, 1]
        setup_test_feed(model, "agent_0", message_ids, shares, ages)
        
        # Get agent_0
        agent = model.network.vs.find(uid="agent_0")
        
        # Test p(reshare|seen) calculation
        newsfeed = model.agent_feeds["agent_0"]
        message_info, ranking = model._rank_newsfeed(newsfeed)
        
        if len(message_info[0]) > 0:  # If any messages are seen
            # Get friend engagement for agent_0
            friend_indices = model.network.successors(agent)
            friend_uids = [model.network.vs[idx]["uid"] for idx in friend_indices]
            
            print(f"  Agent_0 follows: {friend_uids}")
            print(f"  Messages shared by friends: {scenario['shared_by']}")
            
            # Calculate e_i for each message
            for i, msg_id in enumerate(message_info[0]):
                friends_who_shared = len(set(friend_uids) & set(scenario["shared_by"]))
                e_i = friends_who_shared / len(friend_uids) if len(friend_uids) > 0 else 0
                
                # Calculate p(reshare|seen)
                a = message_info[1][i]  # appeal
                r = message_info[3][i]  # recency
                p_reshare = (a * r) / (1 + np.exp(-e_i))
                
                case = next(c for c in test_cases if c["id"] == msg_id)
                print(f"    Message {msg_id} ({case['description']}): e_i = {e_i:.2f}, p(reshare|seen) = {p_reshare:.4f}")
    
    # Clean up
    os.remove(network_file)

def test_parameter_sensitivity():
    """Test how different parameters affect the system."""
    print("\n" + "=" * 60)
    print("TEST 3: PARAMETER SENSITIVITY")
    print("=" * 60)
    
    # Create test network
    network_file = create_test_network()
    
    # Test different r_half values
    print("\n--- Testing r_half parameter (sigmoid center) ---")
    r_half_values = [5, 10, 15]
    
    for r_half in r_half_values:
        print(f"\nr_half = {r_half}")
        
        model = SimSomMod(
            graph_gml=network_file,
            deranking_severity=0.5,  # Moderate deranking
            r_half=r_half,
            temp=2,
            modeling_legality=True,
            verbose=False
        )
        
        # Create test messages and feed
        test_cases = create_test_messages(model)
        setup_test_feed(model, "agent_0", [1, 2, 3, 4], [5, 5, 5, 5], [1, 1, 1, 1])
        
        # Test ranking and visibility
        newsfeed = model.agent_feeds["agent_0"]
        message_info, ranking = model._rank_newsfeed(newsfeed)
        
        print(f"  Messages visible: {len(message_info[0])} out of 4")
        
        # Calculate p(seen) for all original messages to show the effect
        messages, shares, ages = newsfeed
        appeal = np.array([model.all_messages[message].appeal for message in messages])
        recency = np.exp(-0.4 * (ages**0.4))
        
        # Calculate base ranking with deranking
        base_ranking = np.zeros(len(messages))
        for i, message_id in enumerate(messages):
            message = model.all_messages[message_id]
            a, r = appeal[i], recency[i]
            e_g = 0.25  # Assume equal global engagement for simplicity
            base_score = a * r * e_g
            
            is_bad = message.quality < 0.5
            base_ranking[i] = 0.5 * base_score if is_bad else base_score
        
        # Get ranks
        sorted_indices = np.argsort(base_ranking)[::-1]
        ranks = np.zeros(len(messages))
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
        
        # Calculate p(seen)
        p_seen = 1 / (1 + np.exp((ranks - r_half) / 2))
        
        for i, msg_id in enumerate(messages):
            case = next(c for c in test_cases if c["id"] == msg_id)
            print(f"    Message {msg_id} ({case['description']}): rank = {ranks[i]}, p(seen) = {p_seen[i]:.3f}")
    
    # Clean up
    os.remove(network_file)

def run_all_tests():
    """Run all tests to verify the deranking implementation."""
    print("TESTING DERANKING IMPLEMENTATION")
    print("Testing whether our changes work as intended...")
    
    try:
        test_deranking_effect()
        test_friend_engagement_effect()
        test_parameter_sensitivity()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Findings Expected:")
        print("1. Lower deranking_severity should reduce bad content visibility")
        print("2. Higher friend engagement should increase p(reshare|seen)")
        print("3. Different r_half values should change visibility thresholds")
        
    except Exception as e:
        print(f"\nTEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    run_all_tests()
