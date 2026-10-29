[Don't change this document with AI.]


# question

Reliable Retrieval Augmented Generation

Investigate how a language model answers questions using external documents when the retrieved
evidence is incomplete, irrelevant, or contradictory. Develop and evaluate a method to improve evidence
selection, answer faithfulness, or the ability to abstain when evidence is insufficient.

# report

## (1) how RAG respond to insufficient retrieved information
(1) define: 
sufficient context: An instance q′ = (Q, C) has sufficient context if and only if there exists an answer A′ such that A′
is a plausible answer to the question Q given the information in C. Measure with exact match, F1 score, or an LLM-based assessment of answer sameness.

insufficient context (incomplete, irrelevant, or contradictory): not sufficient context

construct dataset with sufficient context autorater 选一个autorater/拿几个比较 section 3.2 & 4.1


(2) contrast experiment, different models given sufficient / insufficient retrived information

compare the rate of correct / hallucinate / abstain

## (2) existing technique of reducing hallucination / increasing abstention (refuse to answer)
combine the sufficient context autorater outputs with model self-rated confidence scores to train a simple linear model to predict hallucinations,
and then use it to set coverage-accuracy trade-off thresholds. section 5.1


## (3) baseline

## (4) new method

## (5) experiment
