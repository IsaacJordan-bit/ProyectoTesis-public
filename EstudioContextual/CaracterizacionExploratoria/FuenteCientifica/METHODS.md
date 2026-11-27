# Exploratory protocol: contextual variation, aggregation and held-out compression

## Scope and status

This protocol is written after the main retrieval results are known. It is
exploratory, not preregistered or an independent replication. The program does not
read those outcomes, change the frozen cohort or choose configurations from
T1/T2/T3. It evaluates the representational rationale of the controlled comparison.

The primary descriptive scope is the frozen 640-word bank, with 80 words in each
of eight operational categories and 100 canonical occurrences per word. A
prespecified sensitivity summary uses the original COMMON subset, without
reselecting words. These categories were built for the retrieval task; they are
not independently annotated sense labels. Fixed categories receive equal weight,
and words receive equal weight within each category.

## Questions and permitted conclusions

1. Is there within-word variation beyond differences between words?
2. How much within-word variation is collapsed by a single mean, and how much is
   represented by the historical support-weighted prototypes?
3. Does a multi-centroid diagnostic describe unseen occurrences better than a
   single training mean, and how does that contrast compare with explicit controls?
4. How stable are those descriptions under changes of source-record sample?
5. When texts are available, does embedding similarity track a separate lexical
   description of the surrounding context?

The geometry question follows the distinction between contextual variation and
anisotropy emphasized by Ethayarajh [1]. The held-out design follows the general
principle of evaluating reproducibility of clustering on observations not used
to fit it [2]; it does **not** compute the published prediction-strength statistic
or use its proposed thresholds. These are purpose-built exploratory diagnostics.

## Source authentication

The original P20 canonical bank also supplies the final P21 classifier extension.
Source files are opened read-only. Before analysis, verify the reviewed bank
protocol and identity, all checksums in its A/P/O receipt, metadata cardinalities,
unique occurrence identities, prototype support, and the identities below.

For word w with n=100 vectors x_i and frozen membership g(i):

\[
a_w = n^{-1}\sum_i x_i,\qquad
p_{wc}=n_{wc}^{-1}\sum_{i:g(i)=c}x_i,\qquad
a_w=\sum_c(n_{wc}/n)p_{wc}.
\]

Agreement uses the historical rtol=atol=1e-12 for these float64 bank arrays.
Metadata are not repaired by sorting to make a failed check pass. Nonfinite
vectors, zero vectors, duplicate identities and missing positions abort the run.

## Descriptive geometry

The total within-word variation, remaining within-prototype variation and
between-prototype variation are:

\[
V_w=\frac1n\sum_i\|x_i-a_w\|^2,\quad
W_w=\frac1n\sum_i\|x_i-p_{w,g(i)}\|^2,\quad
B_w=\sum_c\frac{n_{wc}}n\|p_{wc}-a_w\|^2.
\]

The program verifies V_w=W_w+B_w. The fraction B_w/V_w quantifies how much
within-word variation the frozen prototypes represent. A single constant mean
has zero within-word variation, while keeping every occurrence retains all of it;
these endpoints are algebraic facts, not predictive achievements. Because the
historical prototypes were fitted to these same occurrences, this analysis alone
cannot validate their generalization or semantic interpretation.

Under uniformly random reassignment to fixed nonempty cluster sizes, the expected
between-group fraction is (q-1)/(n-1), where q is the number of prototypes. This
exact finite-sample baseline preserves cluster sizes and is only a descriptive
reference. Beating it is not a significance test: the actual groups were selected
using the vectors, whereas random memberships were not.

For unit vectors u_i=x_i/||x_i||, within-word self-similarity excludes diagonal
pairs. The cross-word reference samples another word uniformly from the remaining
cohort and then its occurrences uniformly. Its exact expectation is computed from
the means of the unit vectors. The difference between self-similarity and this
reference is an anisotropy-adjusted descriptive contrast inspired by [1]; it is
not a replication of that paper's layer-wise experiment.

Additional metrics are unit-direction variance, ||a_w||/mean_i||x_i||, the first
eigenvalue share and entropy effective rank of the **centered** within-word
covariance. The centered first-component fraction here must not be equated to
the paper's maximum-explainable-variance result or its numerical thresholds.
No layer comparison is possible from an already pooled frozen representation.

Nearest-neighbour diagnostics use the 640 frozen word means as a fixed reference
vocabulary and exclude the query's own word. The top-neighbour change rate and
top-10 Jaccard similarity compare each occurrence with its word mean. They measure
variation in neighbourhoods, not improved accuracy, correct senses or new labels.
All these calculations use RAW values; angular diagnostics normalize temporary
copies only. The P20 keynorm retrieval intervention is not reapplied or modified.

## Grouping and held-out evaluation

Within each word, a connected group joins occurrences from the same
`source_doc_key`, exact duplicate embedding vectors, and—if available—identical
context texts after whitespace/case normalization. Each group stays wholly on one
side of a split. `source_doc_key` is a source-record identifier, not evidence of
independence between full documents. Near-duplicate contexts and documents split
across records remain possible dependencies.

Five deterministic repeated splits hold out 20% of groups, rounded upward.
The test occurrence count therefore need not be exactly 20. Nested fractions
0.25, 0.50 and 1.00 of the remaining training groups provide a sampling sensitivity
analysis while holding the test groups fixed. All actual counts are recorded.
At least 10 connected groups, 16 training occurrences, 10 test occurrences and
2k training occurrences are required; unsupported cells remain NOT_EVALUABLE.

For each split and fraction, fit fresh K-means codebooks with k in {1,2,4,8} using
training occurrences only. k=1 is the training mean. The historical A and P,
which incorporate the held-out occurrences, are **never** scored as held-out
estimators. The diagnostic codebooks do not replace the historical P prototypes
and are not a new retrieval condition or an estimate of the number of senses.

For held-out set Q, the reconstruction loss and relative gain are:

\[
L_k=|Q|^{-1}\sum_{x\in Q}\min_{c\leq k}\|x-\hat p_c\|^2,
\qquad G_k=1-L_k/L_1.
\]

The full query vector is available to choose its nearest centroid. Therefore
this quantization task evaluates the ability of a compact codebook to describe
unseen occurrence vectors; it does not demonstrate inference from word identity
alone. Relative gains are averaged within word across repeated splits before
aggregation. A negative gain is retained and interpreted as a failure to improve
over the training mean under that configuration.

The complete set of TRAIN occurrences is also scored as an uncompressed
nearest-occurrence codebook on the same untouched test groups. Its relative gain
against the training mean and its actual number of stored vectors are reported
separately. This is a larger-capacity reference, not a capacity-matched comparison
with one mean or k centroids. Within-word group hold-outs differ from the main
word-disjoint retrieval partitions and do not count as extra retrieval trials.

An exact SVD train-span representation accelerates K-means. All nonzero training
singular directions are retained at numerical rank tolerance. The test component
orthogonal to this span is added back to squared error, so the raw-space loss is
preserved rather than reduced by projection. Test variance outside this span is
also reported, particularly for interpreting the Gaussian reference below.

## Controls and stability

At the full training fraction, every k>1 is compared with 20 random TRAIN
partitions preserving the fitted nonempty cluster sizes. Their centroids are
computed on training data and scored on the same held-out occurrences using the
same nearest-centroid rule. This controls for a codebook of the same size without
vector-informed membership. Empty fitted clusters make that control explicitly
non-evaluable.

A second diagnostic generates 20 pairs of independent Gaussian training/test
samples using the mean and covariance estimated solely from the real training
data. It preserves the empirical covariance eigenvalues and rank, then applies
the same K-means and held-out loss calculations. The real gain is compared with
the mean Gaussian gain. This reference illustrates that continuous unimodal
variation can also benefit from multiple representatives; an excess does not
prove distinct senses or reject every continuous unimodal explanation.

The covariance is an unregularized finite-sample estimate with no variance outside
the training span. This limitation is explicit and can affect contrasts with
real test observations outside that span. Each Gaussian mean has a reported Monte
Carlo standard error. The Gaussian draws are independent and do not preserve dependence between source
records. The 20 draws are computational controls, not 20 empirical
replications or a high-resolution per-word significance test.

For stability, split the training groups into two disjoint halves, fit the same
fixed k in each, and compare their predicted memberships on the **same untouched
test observations** with adjusted Rand index [3]. If either predicted partition
has only one test cluster, mark agreement as degenerate rather than awarding a
perfect stability score. This probe measures sample sensitivity of diagnostic
partitions; it does not certify historical P assignments as linguistic senses.

## Optional lexical-context association

Retrieve only text and identity columns of the canonical occurrence rows. Verify
their identities against the bank, mask case-insensitive exact occurrences of
the target word, aggregate distinct context windows within each connected group,
and compute unigram TF-IDF. English stopwords are removed, terms must occur in at
least two group texts and at most 95%, and vocabulary is capped at 5,000 terms.
At least 20 groups, five terms and nonempty group vectors are required.

Compare upper-triangle cosine similarities of group-mean embeddings with TF-IDF
cosine similarities using Spearman correlation. For 199 controls, permute entire
TF-IDF group identities, applying the same permutation to rows and columns.
Report the observed correlation, permutation mean/quantiles and their difference;
do not treat all pairs as independent observations or report a naive pairwise
p-value. This is a lexical-content proxy, not an independent semantic annotation.

The exact target token is masked, but related inflections, topics, source style
and syntactic information may remain. Both descriptions originate in the same
texts, so association cannot establish causality or linguistic sense validity.
Missing texts are NOT_AVAILABLE; insufficient contexts are NOT_EVALUABLE.
Neither status is converted into a zero correlation or successful validation.

## Weighting, uncertainty and completeness

All empirical summaries give equal weight to words within each category, then
equal weight to categories. Repeated splits are averaged within word first.
The 95% intervals resample words within fixed categories 2,000 times; they are
conditional word-bootstrap sensitivity intervals, not population guarantees for
this nonrandom lexical sample. Shared source records across words are counted
and reported because they limit an independence interpretation of these intervals.

Category omissions recompute the equal-category average after removing one
category, without changing geometry or refitting codebooks. They are descriptive
counterparts to the thesis robustness logic, not new tests of T1/T2/T3. If a
summary lacks every evaluable word in a category, no apparently balanced overall
estimate is produced. Coverage tables retain all planned word/split/fraction/k
cells, and summary endpoints report the number of contributing words.

The program produces manuscript-ready summaries only after all planned words
have checkpoints. Non-evaluable analyses can still exist in a completed run and
remain visible. Software failures stop execution rather than becoming scientific
failures or silently removed words.

## Rationale for the default parameters

| Parameter | Rationale and limitation |
|---|---|
| 640 words / 100 occurrences | Inherited frozen controlled cohort; avoids a new outcome-selected sample. Does not describe the entire original export. |
| k=1,2,4,8 | Coarse geometric resolution at a feasible sample size, including the mean. All sizes are reported; no optimal sense count is claimed. |
| Five split seeds | Repeated source-group sampling for descriptive stability; repeat spread remains available. Five is not evidence of universal convergence. |
| 20% test groups | Reserves unseen source groups while retaining most data for codebooks; actual occurrence counts are reported. |
| 0.25/0.50/1.00 train fractions | Nested sensitivity to the available sample; same test groups and fixed k. |
| Five K-means initializations / 100 iterations | Explicit, bounded computation. Iteration-cap hits and occupied clusters are recorded; disjoint-half stability probes sampling rather than only optimizer restarts. |
| 20 controls | Estimates reference means at bounded cost; per-word Monte Carlo SE is retained. No per-word hypothesis test uses this small ensemble. |
| 199 text permutations | A moderate-cost descriptive reference distribution; not an independently replicated significance claim. |
| 2,000 word bootstraps | Resolves central 95% empirical quantiles; conditional on word-level resampling and the fixed categories. |
| Top-10 neighbours | A fixed, interpretable lexical neighbourhood scale; top-1 changes are also reported. |
| Support thresholds | Avoid codebooks with inadequate training or negligible held-out support. Unsupported cases are disclosed, not reselected. |

All defaults are frozen in PLAN.json before computation. A changed scientific
setting creates a separate run. If further sensitivity analyses are motivated by
these results, describe them as follow-ups rather than replacing inconvenient
results or retroactively calling the original settings preregistered.

## References

[1] K. Ethayarajh, “How contextual are contextualized word representations?
Comparing the geometry of BERT, ELMo, and GPT-2 embeddings,” in *Proc. EMNLP-IJCNLP*,
Hong Kong, China, 2019, pp. 55–65, doi: 10.18653/v1/D19-1006.
https://aclanthology.org/D19-1006/

[2] R. Tibshirani and G. Walther, “Cluster validation by prediction strength,”
*J. Comput. Graph. Stat.*, vol. 14, no. 3, pp. 511–528, 2005,
doi: 10.1198/106186005X59243.
https://doi.org/10.1198/106186005X59243

[3] scikit-learn developers, “adjusted_rand_score,” *scikit-learn documentation*.
Accessed: Nov. 22, 2026. [Online]. Available:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

Implementation documentation also checked: [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
and [NumPy memory-mapped loading](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
The package records actual local versions; it does not require the latest release
or modify the existing P21 environment.
