package scenario;

import java.util.Random;

import org.apache.commons.math3.util.Combinations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import scenario.ipf.IPFAlgorithm;
import scenario.ipf.IPFSampler;
import scenario.ipf.IPFUtils;
import scenario.ipf.problem.IPFProblem;
import scenario.ipf.problem.IPFProblemFromCounts;
import scenario.ipf.problem.IPFProblemFromUnivariateMarginals;

public class RunIPF {
	static public void main(String[] args) {
		INDArray referenceCounts = Nd4j.create(new double[] {
			20.0, 40.0, 40.0, 15.0,
			22.0, 0.0, 38.0, 12.0,
			
			20.0, 40.0, 0.0, 15.0,
			22.0, 45.0, 38.0, 12.0
		}, new int[] { 2, 4, 2 });
		
		//IPFProblem problem = new IPFProblemFromCounts(referenceCounts, new int[] {  2 });
		
		INDArray marginal1 = Nd4j.create(new double[] { 40.0, 60.0 });
		INDArray marginal2 = Nd4j.create(new double[] { 20.0, 30.0, 10.0, 10.0, 10.0, 10.0, 10.0 });
		INDArray marginal3 = Nd4j.create(new double[] { 50.0, 10.0 });
		
		IPFProblem problem = new IPFProblemFromUnivariateMarginals(new INDArray[] { marginal1, marginal2, marginal3 });
		
		IPFAlgorithm algorithm = new IPFAlgorithm(problem);
		
		for (int i = 0; i < 10; i++) {
			algorithm.runOneIteration();

			int[] numberOfCategories = algorithm.getWeights().shape();
			int numberOfDimensions = numberOfCategories.length;

			for (int dimension = 0; dimension < numberOfDimensions; dimension++) {
				double marginal[] = new double[numberOfCategories[dimension]];

				for (int category = 0; category < numberOfCategories[dimension]; category++) {
					INDArray a = algorithm.getWeights().get(
							IPFUtils.getIndices(numberOfDimensions, new int[] { dimension }, new int[] { category }));
					marginal[category] = a.sumNumber().doubleValue();
				}

				System.out.println(Nd4j.create(marginal));
			}
		}
		
		IPFSampler sampler = new IPFSampler(algorithm.getWeights(), new Random(0));
		
		for (int i = 0; i < 1000; i++) {
			int[] sample = sampler.sample();
			
			for (int j = 0; j < sample.length; j++) {
				System.out.print(sample[j] + " ");
			}
			
			System.out.println("");
		}
		

/*		Combinations combinations = new Combinations(4, 2);

		for (int[] combination : combinations) {
			for (int i = 0; i < combination.length; i++)
				System.out.print(combination[i] + " ");
			System.out.println();
		}

		System.exit(1);
		INDArray referenceCounts = Nd4j.create(new double[] { 20.0, 40.0, 40.0, 15.0, 0.0, 45.0, 40.0, 12.0,

				30.0, 50.0, 0.0, 25.0, 28.0, 5.0, 50.0, 22.0,

				25.0, 0.0, 45.0, 20.0, 23.0, 50.0, 45.0, 17.0 }, new int[] { 2, 4, 3 });

		System.out.println(referenceCounts);

		IPFProblem problem = new IPFProblemFromCounts(referenceCounts);
		IPFAlgorithm algorithm = new IPFAlgorithm(problem);

		for (int i = 0; i < 10; i++) {
			algorithm.runOneIteration();

			int[] numberOfCategories = algorithm.getWeights().shape();
			int numberOfDimensions = numberOfCategories.length;

			for (int dimension = 0; dimension < numberOfDimensions; dimension++) {
				double marginal[] = new double[numberOfCategories[dimension]];

				for (int category = 0; category < numberOfCategories[dimension]; category++) {
					INDArray a = algorithm.getWeights().get(
							IPFUtils.getIndices(numberOfDimensions, new int[] { dimension }, new int[] { category }));
					marginal[category] = a.sumNumber().doubleValue();
				}

				System.err.println(Nd4j.create(marginal));
			}
		}*/
	}
}
