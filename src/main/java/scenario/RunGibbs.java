package scenario;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import scenario.gibbs.GibbsAlgorithm;
import scenario.gibbs.GibbsSampler;
import scenario.gibbs.problem.GibbsProblemFromCounts;

public class RunGibbs {
	static public void main(String[] args) {
		INDArray counts = Nd4j.create(new double[] {
				20.0, 40.0, 40.0, 15.0,
				22.0, 0.0, 38.0, 12.0,
				
				20.0, 40.0, 0.0, 15.0,
				22.0, 45.0, 38.0, 12.0
			}, new int[] { 2, 4, 2 });
		
		Random random = new Random(0);
		
		GibbsProblemFromCounts problem = new GibbsProblemFromCounts(counts);
		problem.chooseRandomInitialSample(random);
		
		GibbsAlgorithm algorithm = new GibbsAlgorithm(problem, random);
		GibbsSampler sampler = new GibbsSampler(algorithm, 100, 1000);
		
		for (int i = 0; i < 1000; i++) {
			int[] sample = sampler.sample();
			
			for (int j = 0; j < sample.length; j++) {
				System.out.print(sample[j] + " ");
			}
			
			System.out.println("");
		}
	}
}
