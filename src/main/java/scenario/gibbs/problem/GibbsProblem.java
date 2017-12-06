package scenario.gibbs.problem;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface GibbsProblem {
	INDArray getProbabilities(int dimension, int[] conditionals);
	int[] getInitialSample();
}
