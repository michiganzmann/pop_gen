package scenario.bn;

import java.util.List;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;

public class BNGraphFinder {
	final private BNGraphGenerator generator;
	final private INDArray counts;
	final private List<List<Integer>> data;
	final private Random random;
	
	public BNGraphFinder(BNGraphGenerator generator, INDArray counts, List<List<Integer>> data, Random random) {
		this.generator = generator;
		this.counts = counts;
		this.data = data;
		this.random = random;
	}
	
	public BNGraph findGraph(int numberOfIterations) {
		BNGraph bestGraph = null;
		double bestScore = Double.NEGATIVE_INFINITY;
		
		int messageIter = 0;
		for (int i = 0; i < numberOfIterations; i++) {
			BNGraph graph = generator.generate(data.get(0).size());

			BNProblem problem = new BNProblem(counts);
			BNAlgorithm algorithm = new BNAlgorithm(graph, problem, random);

			double logLikelihood = algorithm.computeLogLikelihood(data);
			double score = logLikelihood;// - algorithm.getNumberOfParameters();
			
			if (score > bestScore) {
				bestGraph = graph;
				bestScore = score;
				
				System.out.println("AIC: " + score + ", Graph: " + bestGraph);
			}
			
			if (messageIter == 25) {
				System.out.println("Iteration: " + i);
				messageIter = 0;
			}
			messageIter++;
		}
		
		return bestGraph;	
	}
}
