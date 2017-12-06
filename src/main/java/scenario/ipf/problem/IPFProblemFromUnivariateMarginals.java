package scenario.ipf.problem;

import org.nd4j.linalg.api.ndarray.INDArray;

public class IPFProblemFromUnivariateMarginals implements IPFProblem {
	final private INDArray[] marginals;
	final private int[] shape;
	
	public IPFProblemFromUnivariateMarginals(INDArray[] marginals) {
		this.marginals = marginals;
		this.shape = new int[marginals.length];
		
		int i = 0;
		for (INDArray marginal : marginals) {
			if (marginal.shape().length > 2) {
				throw new IllegalStateException();
			}
			
			this.shape[i] = marginal.shape()[1];
			i++;
		}
	}
	
	@Override
	public Number getMarginalCounts(int[] dimensions, int[] categories) {
		if (dimensions.length > 1 || categories.length > 1) {
			throw new IllegalStateException();
		}
		
		return marginals[dimensions[0]].getDouble(categories[0]);
	}

	@Override
	public int[] getShape() {
		return shape;
	}

	@Override
	public int[] getLevels() {
		return new int[] { 1 };
	}

}
