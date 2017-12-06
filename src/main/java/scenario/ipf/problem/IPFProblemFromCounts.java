package scenario.ipf.problem;

import org.nd4j.linalg.api.ndarray.INDArray;

import scenario.ipf.IPFUtils;

public class IPFProblemFromCounts implements IPFProblem {
	final private INDArray referenceCounts;
	final private int numberOfDimensions;
	final private int[] levels;

	public IPFProblemFromCounts(INDArray referenceCounts, int[] levels) {
		this.referenceCounts = referenceCounts;
		this.numberOfDimensions = referenceCounts.shape().length;
		this.levels = levels;
	}

	@Override
	public Number getMarginalCounts(int[] dimensions, int[] categories) {
		return referenceCounts.get(IPFUtils.getIndices(numberOfDimensions, dimensions, categories)).sumNumber();
	}

	@Override
	public int[] getShape() {
		return referenceCounts.shape();
	}

	@Override
	public int[] getLevels() {
		return levels;
	}
}
