import numpy as np

class ClassificationMetrics:
    """Metrics for binary and multi-class classification"""
    
    @staticmethod
    def precision(y_true, y_pred, average='binary', pos_label=1):
        """
        Precision = TP / (TP + FP)
        """
        if average == 'binary':
            tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
            fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        elif average == 'macro':
            classes = np.unique(y_true)
            precisions = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fp = np.sum((y_true != cls) & (y_pred == cls))
                precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            return np.mean(precisions)
        
        elif average == 'micro':
            tp = np.sum(y_true == y_pred)
            fp = np.sum(y_true != y_pred)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @staticmethod
    def recall(y_true, y_pred, average='binary', pos_label=1):
        """
        Recall = TP / (TP + FN)
        """
        if average == 'binary':
            tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
            fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        elif average == 'macro':
            classes = np.unique(y_true)
            recalls = []
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))
                recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            return np.mean(recalls)
        
        elif average == 'micro':
            tp = np.sum(y_true == y_pred)
            fn = np.sum(y_true != y_pred)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @staticmethod
    def f1_score(y_true, y_pred, average='binary', pos_label=1):
        """
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        prec = ClassificationMetrics.precision(y_true, y_pred, average, pos_label)
        rec = ClassificationMetrics.recall(y_true, y_pred, average, pos_label)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    @staticmethod
    def roc_curve(y_true, y_scores):
        """
        Compute ROC curve: TPR vs FPR at different thresholds
        Returns: (fpr, tpr, thresholds)
        """
        # Sort by descending scores
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        y_scores_sorted = y_scores[desc_score_indices]
        
        # Get unique thresholds
        thresholds = np.unique(y_scores_sorted)
        thresholds = np.concatenate([thresholds, [thresholds[-1] - 1]])
        
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            # TPR = TP / (TP + FN)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # FPR = FP / (FP + TN)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        return np.array(fpr_list), np.array(tpr_list), thresholds
    
    @staticmethod
    def auc(y_true, y_scores):
        """
        Compute Area Under ROC Curve using trapezoidal rule
        AUC = ∫ TPR d(FPR)
        """
        fpr, tpr, _ = ClassificationMetrics.roc_curve(y_true, y_scores)
        
        # Sort by fpr to ensure proper integration
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]
        
        # Trapezoidal rule: sum of (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
        auc_value = 0.0
        for i in range(len(fpr_sorted) - 1):
            auc_value += (fpr_sorted[i + 1] - fpr_sorted[i]) * (tpr_sorted[i + 1] + tpr_sorted[i]) / 2
        
        return auc_value


class RankingMetrics:
    """Metrics for ranking and recommendation systems"""
    
    @staticmethod
    def pass_at_k(generated_samples, reference, k):
        """
        Pass@k: Probability that at least one of top-k samples is correct
        For code generation: does at least 1 of k attempts pass tests?
        
        Args:
            generated_samples: list of generated outputs
            reference: correct answer or test function
            k: number of samples to consider
        Returns:
            pass_rate: float between 0 and 1
        """
        if len(generated_samples) < k:
            k = len(generated_samples)
        
        # Check if any of top k samples match reference
        top_k_samples = generated_samples[:k]
        
        if callable(reference):
            # reference is a test function
            passed = any(reference(sample) for sample in top_k_samples)
        else:
            # reference is the correct answer
            passed = any(sample == reference for sample in top_k_samples)
        
        return 1.0 if passed else 0.0
    
    @staticmethod
    def hit_at_k(ranked_items, relevant_items, k):
        """
        Hit@k: Whether any relevant item appears in top-k
        Binary metric: 1 if at least one relevant item in top-k, else 0
        
        Args:
            ranked_items: list of items in ranked order
            relevant_items: set of relevant items
            k: cutoff position
        Returns:
            hit: 1 or 0
        """
        if len(ranked_items) < k:
            k = len(ranked_items)
        
        top_k = set(ranked_items[:k])
        relevant_set = set(relevant_items)
        
        return 1 if len(top_k & relevant_set) > 0 else 0
    
    @staticmethod
    def hit_rate_at_k(ranked_lists, relevant_lists, k):
        """
        Average Hit@k over multiple queries
        
        Args:
            ranked_lists: list of ranked item lists (one per query)
            relevant_lists: list of relevant item sets (one per query)
            k: cutoff position
        Returns:
            average hit rate
        """
        hits = []
        for ranked_items, relevant_items in zip(ranked_lists, relevant_lists):
            hits.append(RankingMetrics.hit_at_k(ranked_items, relevant_items, k))
        return np.mean(hits)
    
    @staticmethod
    def average_precision(ranked_items, relevant_items, k=None):
        """
        Average Precision: average of precision values at positions where relevant items occur
        AP = (1/|relevant|) * Σ P(k) * rel(k)
        
        Args:
            ranked_items: list of items in ranked order
            relevant_items: set of relevant items
            k: cutoff position (None = use all items)
        Returns:
            average precision score
        """
        if k is None:
            k = len(ranked_items)
        
        relevant_set = set(relevant_items)
        if len(relevant_set) == 0:
            return 0.0
        
        num_hits = 0
        sum_precisions = 0.0
        
        for i in range(min(k, len(ranked_items))):
            if ranked_items[i] in relevant_set:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / len(relevant_set)
    
    @staticmethod
    def mean_average_precision(ranked_lists, relevant_lists, k=None):
        """
        MAP: Mean of Average Precision across multiple queries
        MAP = (1/|Q|) * Σ AP(q)
        
        Args:
            ranked_lists: list of ranked item lists (one per query)
            relevant_lists: list of relevant item sets (one per query)
            k: cutoff position (None = use all items)
        Returns:
            mean average precision
        """
        aps = []
        for ranked_items, relevant_items in zip(ranked_lists, relevant_lists):
            ap = RankingMetrics.average_precision(ranked_items, relevant_items, k)
            aps.append(ap)
        return np.mean(aps)
    
    @staticmethod
    def dcg_at_k(ranked_items, relevance_scores, k=None):
        """
        Discounted Cumulative Gain
        DCG = Σ (rel_i / log2(i + 1))
        
        Args:
            ranked_items: list of items in ranked order
            relevance_scores: dict mapping items to relevance scores
            k: cutoff position (None = use all items)
        Returns:
            DCG score
        """
        if k is None:
            k = len(ranked_items)
        
        dcg = 0.0
        for i in range(min(k, len(ranked_items))):
            item = ranked_items[i]
            relevance = relevance_scores.get(item, 0)
            # Position is i+1 (1-indexed), so log2(i+2) for denominator
            dcg += relevance / np.log2(i + 2)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(ranked_items, relevance_scores, k=None):
        """
        Normalized Discounted Cumulative Gain
        NDCG = DCG / IDCG
        where IDCG is DCG of ideal ranking
        
        Args:
            ranked_items: list of items in ranked order
            relevance_scores: dict mapping items to relevance scores
            k: cutoff position (None = use all items)
        Returns:
            NDCG score (0 to 1)
        """
        if k is None:
            k = len(ranked_items)
        
        # Compute DCG for actual ranking
        dcg = RankingMetrics.dcg_at_k(ranked_items, relevance_scores, k)
        
        # Compute IDCG: sort items by relevance (descending)
        all_items = list(relevance_scores.keys())
        ideal_ranking = sorted(all_items, key=lambda x: relevance_scores[x], reverse=True)
        idcg = RankingMetrics.dcg_at_k(ideal_ranking, relevance_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_ndcg_at_k(ranked_lists, relevance_score_lists, k=None):
        """
        Average NDCG across multiple queries
        
        Args:
            ranked_lists: list of ranked item lists (one per query)
            relevance_score_lists: list of relevance score dicts (one per query)
            k: cutoff position (None = use all items)
        Returns:
            mean NDCG score
        """
        ndcgs = []
        for ranked_items, relevance_scores in zip(ranked_lists, relevance_score_lists):
            ndcg = RankingMetrics.ndcg_at_k(ranked_items, relevance_scores, k)
            ndcgs.append(ndcg)
        return np.mean(ndcgs)


# ============================================================
# Test Cases
# ============================================================

print("=" * 60)
print("Test Case 1: Binary Classification Metrics")
print("=" * 60)

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])

prec = ClassificationMetrics.precision(y_true, y_pred)
rec = ClassificationMetrics.recall(y_true, y_pred)
f1 = ClassificationMetrics.f1_score(y_true, y_pred)

print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")


print("\n" + "=" * 60)
print("Test Case 2: Multi-class Classification (Macro Average)")
print("=" * 60)

y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_pred_multi = np.array([0, 1, 2, 0, 2, 2, 1, 1, 2, 0])

prec_macro = ClassificationMetrics.precision(y_true_multi, y_pred_multi, average='macro')
rec_macro = ClassificationMetrics.recall(y_true_multi, y_pred_multi, average='macro')
f1_macro = ClassificationMetrics.f1_score(y_true_multi, y_pred_multi, average='macro')

print(f"y_true: {y_true_multi}")
print(f"y_pred: {y_pred_multi}")
print(f"Precision (macro): {prec_macro:.3f}")
print(f"Recall (macro): {rec_macro:.3f}")
print(f"F1 Score (macro): {f1_macro:.3f}")


print("\n" + "=" * 60)
print("Test Case 3: ROC Curve and AUC")
print("=" * 60)

y_true_roc = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
y_scores = np.array([0.1, 0.2, 0.35, 0.8, 0.3, 0.9, 0.15, 0.7, 0.85, 0.95])

fpr, tpr, thresholds = ClassificationMetrics.roc_curve(y_true_roc, y_scores)
auc_value = ClassificationMetrics.auc(y_true_roc, y_scores)

print(f"y_true: {y_true_roc}")
print(f"y_scores: {y_scores}")
print(f"\nROC Curve Points (first 5):")
for i in range(min(5, len(fpr))):
    print(f"  Threshold: {thresholds[i]:.2f}, FPR: {fpr[i]:.3f}, TPR: {tpr[i]:.3f}")
print(f"\nAUC Score: {auc_value:.3f}")


print("\n" + "=" * 60)
print("Test Case 4: Pass@k for Code Generation")
print("=" * 60)

# Simulate code generation: 5 attempts, correct answer is "def add(a,b): return a+b"
generated_code = [
    "def add(a,b): return a-b",  # wrong
    "def add(a,b): return a*b",  # wrong
    "def add(a,b): return a+b",  # correct
    "def add(a,b): return b+a",  # also correct
    "def add(a,b): return sum([a,b])"  # also correct
]

correct_answer = "def add(a,b): return a+b"

pass_at_1 = RankingMetrics.pass_at_k(generated_code, correct_answer, k=1)
pass_at_3 = RankingMetrics.pass_at_k(generated_code, correct_answer, k=3)
pass_at_5 = RankingMetrics.pass_at_k(generated_code, correct_answer, k=5)

print(f"Generated code attempts: {len(generated_code)}")
print(f"Pass@1: {pass_at_1:.2f}")
print(f"Pass@3: {pass_at_3:.2f}")
print(f"Pass@5: {pass_at_5:.2f}")


print("\n" + "=" * 60)
print("Test Case 5: Hit@k for Recommendation")
print("=" * 60)

# User likes items [5, 8, 12]
# System recommends items in order: [3, 5, 7, 8, 10]
ranked_items = [3, 5, 7, 8, 10]
relevant_items = [5, 8, 12]

hit_at_1 = RankingMetrics.hit_at_k(ranked_items, relevant_items, k=1)
hit_at_3 = RankingMetrics.hit_at_k(ranked_items, relevant_items, k=3)
hit_at_5 = RankingMetrics.hit_at_k(ranked_items, relevant_items, k=5)

print(f"Ranked items: {ranked_items}")
print(f"Relevant items: {relevant_items}")
print(f"Hit@1: {hit_at_1}")
print(f"Hit@3: {hit_at_3}")
print(f"Hit@5: {hit_at_5}")


print("\n" + "=" * 60)
print("Test Case 6: Hit Rate@k (Multiple Queries)")
print("=" * 60)

# 3 different queries
ranked_lists = [
    [1, 2, 3, 4, 5],
    [10, 20, 30, 40, 50],
    [100, 200, 300, 400, 500]
]
relevant_lists = [
    [3, 6, 9],
    [20, 25],
    [150, 250]
]

hit_rate_at_3 = RankingMetrics.hit_rate_at_k(ranked_lists, relevant_lists, k=3)
hit_rate_at_5 = RankingMetrics.hit_rate_at_k(ranked_lists, relevant_lists, k=5)

print(f"Number of queries: {len(ranked_lists)}")
print(f"Hit Rate@3: {hit_rate_at_3:.3f}")
print(f"Hit Rate@5: {hit_rate_at_5:.3f}")


print("\n" + "=" * 60)
print("Test Case 7: Average Precision (AP)")
print("=" * 60)

# System returns: [doc1, doc2, doc3, doc4, doc5]
# Relevant docs: [doc2, doc3, doc5]
ranked_docs = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
relevant_docs = ['doc2', 'doc3', 'doc5']

ap = RankingMetrics.average_precision(ranked_docs, relevant_docs)

print(f"Ranked documents: {ranked_docs}")
print(f"Relevant documents: {relevant_docs}")
print(f"Average Precision: {ap:.3f}")
print(f"\nManual calculation:")
print(f"  doc2 at position 2: P@2 = 1/2 = 0.500")
print(f"  doc3 at position 3: P@3 = 2/3 = 0.667")
print(f"  doc5 at position 5: P@5 = 3/5 = 0.600")
print(f"  AP = (0.500 + 0.667 + 0.600) / 3 = {ap:.3f}")


print("\n" + "=" * 60)
print("Test Case 8: Mean Average Precision (MAP)")
print("=" * 60)

# Multiple queries
ranked_lists_map = [
    ['doc1', 'doc2', 'doc3', 'doc4'],
    ['item_a', 'item_b', 'item_c', 'item_d'],
    ['A', 'B', 'C', 'D', 'E']
]
relevant_lists_map = [
    ['doc2', 'doc4'],
    ['item_a', 'item_c'],
    ['B', 'D']
]

map_score = RankingMetrics.mean_average_precision(ranked_lists_map, relevant_lists_map)
map_at_3 = RankingMetrics.mean_average_precision(ranked_lists_map, relevant_lists_map, k=3)

print(f"Number of queries: {len(ranked_lists_map)}")
print(f"MAP (all positions): {map_score:.3f}")
print(f"MAP@3: {map_at_3:.3f}")


print("\n" + "=" * 60)
print("Test Case 9: DCG and NDCG")
print("=" * 60)

# Search results with graded relevance (0=not relevant, 1=somewhat, 2=relevant, 3=highly relevant)
ranked_results = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
relevance_scores = {
    'doc1': 3,
    'doc2': 2,
    'doc3': 3,
    'doc4': 0,
    'doc5': 1
}

dcg_5 = RankingMetrics.dcg_at_k(ranked_results, relevance_scores, k=5)
ndcg_5 = RankingMetrics.ndcg_at_k(ranked_results, relevance_scores, k=5)

# Ideal ranking would be: doc1(3), doc3(3), doc2(2), doc5(1), doc4(0)
ideal_ranking = ['doc1', 'doc3', 'doc2', 'doc5', 'doc4']
idcg_5 = RankingMetrics.dcg_at_k(ideal_ranking, relevance_scores, k=5)

print(f"Ranked results: {ranked_results}")
print(f"Relevance scores: {relevance_scores}")
print(f"\nDCG@5: {dcg_5:.3f}")
print(f"IDCG@5: {idcg_5:.3f}")
print(f"NDCG@5: {ndcg_5:.3f}")


print("\n" + "=" * 60)
print("Test Case 10: Mean NDCG (Multiple Queries)")
print("=" * 60)

ranked_lists_ndcg = [
    ['A', 'B', 'C', 'D'],
    ['item1', 'item2', 'item3', 'item4'],
    ['x', 'y', 'z']
]
relevance_score_lists = [
    {'A': 3, 'B': 2, 'C': 1, 'D': 0},
    {'item1': 2, 'item2': 3, 'item3': 1, 'item4': 2},
    {'x': 1, 'y': 3, 'z': 2}
]

mean_ndcg = RankingMetrics.mean_ndcg_at_k(ranked_lists_ndcg, relevance_score_lists)
mean_ndcg_at_3 = RankingMetrics.mean_ndcg_at_k(ranked_lists_ndcg, relevance_score_lists, k=3)

print(f"Number of queries: {len(ranked_lists_ndcg)}")
print(f"Mean NDCG (all): {mean_ndcg:.3f}")
print(f"Mean NDCG@3: {mean_ndcg_at_3:.3f}")
