package livewire;

import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_highgui;

import java.util.*;

/**
 * WRITE CLASS DESCRIPTION HERE
 */
public class CostMap
{
    private static final float RAD2 = 1.41421356f;
    Node[][] original;
    Node[][] costs;
    Node current;

    public class Node
    {
        public short row;
        public short col;
        public int cost;
        public Node parent;

        public boolean equals(Node n) {
            return n.row == this.row && n.col == this.col;
        }

        @Override
        public String toString() {
            String parent = "set";
            if (parent == null)
                parent = "null";

            return "Node[row:" +row + " col:" + col + " cost:" + cost +
                    " parent:" + parent + "]";
        }
    }


    public CostMap(CvMat image) {
        reset(image);
    }

    public void reset(CvMat image) {
        original = new Node[image.rows()][image.cols()];
        costs = new Node[image.rows()][image.cols()];

        //keep a copy of the original costs
        for (int i = 0; i < original.length; i++) {
            for (int j = 0; j < original[0].length; j++) {
                Node n = new Node();
                n.row = (short) i;
                n.col = (short) j;
                n.cost = (int) image.get(i, j);
                n.parent = null;
                original[i][j] = n;
            }
        }

        reset();
    }

    public void reset() {
        //make a deep copy of the original array of nodes
        for (int i = 0; i < original.length; i++) {
            for (int j = 0; j < original[0].length; j++) {
                Node n = new Node();
                n.row = original[i][j].row;
                n.col = original[i][j].col;
                n.cost = original[i][j].cost;
                n.parent = null;
                costs[i][j] = n;
            }
        }
    }

    private enum ExpandMethod { COST, DIST }

    public Node addSeed(int row, int col) {
        //TODO keep track of seed points for path cooling
        System.out.println("New seed-point row:" + row + " col:" + col);
        reset();
        expand(row, col, ExpandMethod.COST);

        return costs[row][col];
    }

    public Node getNode(int row, int col){
        return costs[row][col];
    }

    private void expand(int row, int col, ExpandMethod method) {
        //create algorithm data structures COMPARE VIA CUM COST
        Set<Node> closed = new HashSet<Node>();
        PriorityQueue<Node> wavefront;
        //use specified comparator
        switch (method) {
            case COST:
                wavefront = new PriorityQueue<Node>(1000, new Comparator<Node>()
                {
                    @Override
                    public int compare(Node a, Node b) {
                        if (a.cost > b.cost) return 1;
                        if (a.cost < b.cost) return -1;
                        return 0;
                    }
                });
                break;
            case DIST:
                wavefront = new PriorityQueue<Node>(1000, new Comparator<Node>()
                {
                    @Override
                    public int compare(Node a, Node b) {
                        double distA = euclideanDist(current, a);
                        double distB = euclideanDist(current, b);
                        if (distA > distB)
                            return 1;
                        if (distA < distB) return -1;
                        return 0;
                    }
                });
                break;
            default:
                System.out.println("Invalid method argument for CostMap.expand()");
                return;
        }

        //get seed point, initialize cost to 0, and add to wavefront
        current = costs[row][col];
        current.cost = 0;
        wavefront.add(current);

        int count = 0;
        float size = (float) original.length * original[0].length;
        int step = (int) size / 20;
        System.out.println("Expanding graph...");
        while (!wavefront.isEmpty()) {
            //get next lowest cost Node from wavefront and add to closed set
            current = wavefront.poll();
            closed.add(current);

            //get neighbors of current and expandVcost
            ArrayList<Node> neigbors = getNeighbors(current);
            for (int i = 0; i < neigbors.size(); i++) {
                Node n = neigbors.get(i);
                if (closed.contains(n)) continue;

                int tentativeCost = euclideanAdd(current, n);
                if (n.parent == null || current.cost + tentativeCost < n.cost) {
                    n.parent = current;
                    n.cost = tentativeCost;
                }

                //add neighbors to wavefront if not already in
                if (!wavefront.contains(n)) {
                    wavefront.add(n);
                }
            }

            if (count % step == 0) {
                System.out.println("Expanding:  " + (int) (100 * (count / size)) +
                        "%");
            }
            count++;
        }
        System.out.println("Expanding: 100%");
    }

    private ArrayList<Node> getNeighbors(Node n) {
        ArrayList<Node> neighbors = new ArrayList<Node>(8);

        for (int i = n.row - 1; i <= n.row + 1; i++) {
            for (int j = n.col - 1; j <= n.col + 1; j++) {
                //skip node n
                if (i == n.row && j == n.col) continue;
                //check if current neighbor is within image bounds
                if (i >= 0 && j >= 0 && i < costs.length && j < costs[0].length) {
                    neighbors.add(costs[i][j]);
                }
            }
        }

        return neighbors;
    }

    private int euclideanAdd(Node current, Node n) {
        //if diagonal, scale by RAD2
        if (current.row != n.row && current.col != n.col)
            return current.cost + (int) (RAD2 * costs[n.row][n.col].cost);

        return current.cost + costs[n.row][n.col].cost;
    }

    private double euclideanDist(Node a, Node b) {
        return (double) Math.sqrt(((b.row - a.row) * (b.row - a.row)) +
                ((b.col - a.col) * (b.col - a.col)));
    }

}