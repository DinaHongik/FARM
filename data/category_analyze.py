"""
Analyze IFTTT Dataset by Categories
Shows if category-based grouping can solve imbalance problem
"""

import json
from collections import defaultdict, Counter


def analyze_categories(input_path='ifttt_applets_complete.json'):
    """
    Analyze trigger and action categories to see if grouping
    by category reduces the imbalance problem
    """
    
    print("=" * 80)
    print("IFTTT CATEGORY-BASED ANALYSIS")
    print("=" * 80)
    print(f"Input: {input_path}\n")
    
    # Load dataset
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Statistics
    total_applets = 0
    trigger_categories = Counter()
    action_categories = Counter()
    category_pair_counts = defaultdict(int)
    category_pair_to_applets = defaultdict(list)
    service_pair_counts = defaultdict(int)
    cross_category_count = 0
    
    # Parse structure
    services = data.get('root', data) if isinstance(data, dict) else data
    
    print("Parsing categories...")
    
    for service_idx, service in enumerate(services):
        if not isinstance(service, dict):
            continue
        
        if (service_idx + 1) % 100 == 0:
            print(f"  Processing service {service_idx + 1}/{len(services)}...", end='\r')
        
        applets = service.get('applets', [])
        
        for applet in applets:
            if not isinstance(applet, dict):
                continue
            
            total_applets += 1
            
            # Get categories
            trig_cats = applet.get('trigger_categories', [])
            act_cats = applet.get('action_categories', [])
            is_cross = applet.get('cross_category', False)
            
            if is_cross:
                cross_category_count += 1
            
            # Extract trigger and action services
            components = applet.get('components', [])
            if not isinstance(components, list) or len(components) < 2:
                continue
            
            trigger = None
            action = None
            
            for comp in components:
                if not isinstance(comp, dict):
                    continue
                label = comp.get('label', '').lower()
                if label == 'if':
                    trigger = comp.get('service_name', 'Unknown')
                elif label == 'then':
                    action = comp.get('service_name', 'Unknown')
            
            if not trigger or not action:
                continue
            
            # Track service-level pairs
            service_pair_counts[(trigger, action)] += 1
            
            # Track categories
            if trig_cats:
                for tc in trig_cats:
                    trigger_categories[tc] += 1
            
            if act_cats:
                for ac in act_cats:
                    action_categories[ac] += 1
            
            # Track category pairs (use first category of each)
            trig_cat = trig_cats[0] if trig_cats else "Uncategorized"
            act_cat = act_cats[0] if act_cats else "Uncategorized"
            
            cat_pair = (trig_cat, act_cat)
            category_pair_counts[cat_pair] += 1
            category_pair_to_applets[cat_pair].append({
                'trigger': trigger,
                'action': action,
                'description': applet.get('description', '')
            })
    
    print(" " * 80, end='\r')
    print(f"Processed {total_applets:,} applets\n")
    
    # Analysis
    print("=" * 80)
    print("COMPARISON: SERVICE-LEVEL vs CATEGORY-LEVEL")
    print("=" * 80 + "\n")
    
    print(f"Service-level (current approach):")
    print(f"  Unique trigger services: {len(set(p[0] for p in service_pair_counts))}")
    print(f"  Unique action services: {len(set(p[1] for p in service_pair_counts))}")
    print(f"  Unique pairs: {len(service_pair_counts):,}")
    print()
    
    print(f"Category-level (proposed approach):")
    print(f"  Unique trigger categories: {len(trigger_categories)}")
    print(f"  Unique action categories: {len(action_categories)}")
    print(f"  Unique category pairs: {len(category_pair_counts):,}")
    print(f"  Cross-category applets: {cross_category_count:,} ({cross_category_count/total_applets*100:.1f}%)")
    print()
    
    reduction = (1 - len(category_pair_counts) / len(service_pair_counts)) * 100
    print(f"Complexity reduction: {reduction:.1f}%")
    print()
    
    # Category distribution
    print("=" * 80)
    print("TRIGGER CATEGORIES (Top 20)")
    print("=" * 80 + "\n")
    
    for i, (cat, count) in enumerate(trigger_categories.most_common(20), 1):
        print(f"{i:2}. {cat:40} {count:6,} applets ({count/total_applets*100:5.1f}%)")
    
    print(f"\nTotal trigger categories: {len(trigger_categories)}")
    
    print("\n" + "=" * 80)
    print("ACTION CATEGORIES (Top 20)")
    print("=" * 80 + "\n")
    
    for i, (cat, count) in enumerate(action_categories.most_common(20), 1):
        print(f"{i:2}. {cat:40} {count:6,} applets ({count/total_applets*100:5.1f}%)")
    
    print(f"\nTotal action categories: {len(action_categories)}")
    
    # Category pair distribution
    print("\n" + "=" * 80)
    print("CATEGORY PAIR FREQUENCY DISTRIBUTION")
    print("=" * 80 + "\n")
    
    ranges = [
        (1, 1, "1 example"),
        (2, 4, "2-4 examples"),
        (5, 9, "5-9 examples"),
        (10, 19, "10-19 examples"),
        (20, 49, "20-49 examples"),
        (50, 99, "50-99 examples"),
        (100, 499, "100-499 examples"),
        (500, float('inf'), "500+ examples")
    ]
    
    for min_count, max_count, label in ranges:
        pairs_in_range = sum(1 for count in category_pair_counts.values() 
                            if min_count <= count <= max_count)
        applets_in_range = sum(count for count in category_pair_counts.values() 
                              if min_count <= count <= max_count)
        
        if pairs_in_range > 0:
            print(f"  {label:20} {pairs_in_range:4} pairs ({pairs_in_range/len(category_pair_counts)*100:5.1f}%)  "
                  f"{applets_in_range:6,} applets ({applets_in_range/total_applets*100:5.1f}%)")
    
    # Top category pairs
    print("\n" + "=" * 80)
    print("TOP 30 CATEGORY PAIRS")
    print("=" * 80 + "\n")
    
    sorted_cat_pairs = sorted(category_pair_counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, ((trig_cat, act_cat), count) in enumerate(sorted_cat_pairs[:30], 1):
        print(f"{i:2}. {trig_cat} -> {act_cat}")
        print(f"    {count:,} applets")
        
        # Show sample service pairs
        sample = category_pair_to_applets[(trig_cat, act_cat)][:2]
        for s in sample:
            print(f"       - {s['trigger']} -> {s['action']}")
    
    # Key thresholds
    print("\n" + "=" * 80)
    print("CATEGORY-BASED TRAINING FEASIBILITY")
    print("=" * 80 + "\n")
    
    for threshold in [5, 10, 20, 50, 100]:
        eligible = sum(1 for count in category_pair_counts.values() if count >= threshold)
        eligible_applets = sum(count for count in category_pair_counts.values() if count >= threshold)
        
        print(f"Category pairs with >={threshold} examples:")
        print(f"  Pairs: {eligible} ({eligible/len(category_pair_counts)*100:.1f}%)")
        print(f"  Applets: {eligible_applets:,} ({eligible_applets/total_applets*100:.1f}%)")
        print()
    
    # Final recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80 + "\n")
    
    pairs_50plus = sum(1 for count in category_pair_counts.values() if count >= 50)
    
    if pairs_50plus >= 50:
        print("CATEGORY-BASED APPROACH IS VIABLE")
        print(f"\nYou have {pairs_50plus} category pairs with >=50 examples")
        print(f"This is {reduction:.0f}% reduction in complexity vs service-level")
        print("\nStrategy:")
        print("  1. Train on category pairs (high-level)")
        print("  2. Use few-shot prompting for specific services")
        print("  3. Much better data balance than service-level")
    else:
        print("CATEGORY-BASED APPROACH HAS LIMITATIONS")
        print(f"\nYou only have {pairs_50plus} category pairs with >=50 examples")
        print("\nConsider:")
        print("  1. Hybrid: categories for triggers, services for actions")
        print("  2. Or stick with top service pairs")
    
    # Save analysis
    output = {
        'summary': {
            'total_applets': total_applets,
            'service_level_pairs': len(service_pair_counts),
            'category_level_pairs': len(category_pair_counts),
            'complexity_reduction_pct': reduction,
            'unique_trigger_categories': len(trigger_categories),
            'unique_action_categories': len(action_categories)
        },
        'top_trigger_categories': [
            {'category': cat, 'count': count} 
            for cat, count in trigger_categories.most_common(30)
        ],
        'top_action_categories': [
            {'category': cat, 'count': count}
            for cat, count in action_categories.most_common(30)
        ],
        'top_category_pairs': [
            {
                'trigger_category': trig,
                'action_category': act,
                'count': count,
                'sample_services': category_pair_to_applets[(trig, act)][:5]
            }
            for (trig, act), count in sorted_cat_pairs[:50]
        ]
    }
    
    output_path = 'category_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    analyze_categories('ifttt_applets_complete.json')