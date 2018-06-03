package scenario;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;

import scenario.bn.BNAlgorithm;
import scenario.bn.BNGraph;
import scenario.bn.BNGraphFinder;
import scenario.bn.BNGraphGenerator;
import scenario.bn.BNProblem;
import scenario.data.CSVReader;
import scenario.data.CountsGenerator;

public class TestNetworkChoice {
	static public void main(String[] args) throws IOException {
		// CSV is sorted by kommas ","
		List<List<Integer>> data = new CSVReader(",").load(new File("data/haushaltData.csv"),
				Arrays.asList("f30100","f32200a","F20601","hhgr","minAlter","maxAlter"));
		// 0: Autos, 1: Velos, 2: Haushaltseinkommen, 3: Haushaltsgrösse, 4: minAlter, 5: maxAlter
		System.out.println("done CSV");
		Collections.shuffle(data);
		List<List<Integer>> reducedData = data.subList(0, 4000);
		System.out.println("done reduced Data");

		INDArray counts = new CountsGenerator().getCounts(data);
		System.out.println("done counts");
		INDArray reducedCounts = new CountsGenerator().getCounts(reducedData, data);
		System.out.println("done reducedCounts");

		Random random = new Random(0);
		BNGraphGenerator graphGenerator = new BNGraphGenerator(random);
		System.out.println("done graphGenerator");
		
		BNGraphFinder graphFinder = new BNGraphFinder(graphGenerator, reducedCounts, reducedData, random);
		System.out.println("done GraphFinder");
		BNGraph graph = graphFinder.findGraph(500);
		System.out.println("done Graph");
	}
}
