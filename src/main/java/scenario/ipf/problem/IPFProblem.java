package scenario.ipf.problem;

public interface IPFProblem {
	Number getMarginalCounts(int[] dimensions, int[] categories);
	int[] getShape();
	int[] getLevels();
}
