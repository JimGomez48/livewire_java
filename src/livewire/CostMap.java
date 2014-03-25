package livewire;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvPoint;

import java.util.*;


/**
 * CostMap is used to calculate and store the cumulative costs of the image pixels
 * from a given seedpoint. It uses a variation of Dijkstra's shortest path algorithm
 * to expand the image pixel graph and calculate least cost paths from the seed point
 * to all other pixels in the image.
 *
 * @author James Gomez
 */
public class CostMap
{
    private static final float RAD2 = 1.41421356f;
    Node[][] original;
    Node[][] costs;
    Node lastSeed;

    /**
     * Used for the expansion algorithm to store cumulative costs and parent
     * pointers
     */
    public class Node
    {
        /**
         * the row of the pixel that this node corresponds to
         */
        public short row;
        /**
         * the column of the pixel that this node corresponds to
         */
        public short col;
        /**
         * the current cumulative cost of this node from the seed point
         */
        public int cost;
        /**
         * the next node along the lowest cost path from this node to the seed
         */
        public Node parent;

        public boolean equals(Node n) {
            return n.row == this.row && n.col == this.col;
        }

        @Override
        public String toString() {
            String parent = "set";
            if (parent == null)
                parent = "null";

            return "Node(row:" + row + " col:" + col + " cost:" + cost +
                    " parent:" + parent + ")\n";
        }
    }

    /**
     * Creates a CostMap with the given CvMat image as its starting data
     */
    public CostMap(CvMat image) {
        reset(image);
    }

    /**
     * Resets the algorithm to a non-expanded state using a new image
     */
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

    /**
     * Resets the algorithm to the original, non-expanded state.
     */
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

    /**
     * Sets a seedpoint as the starting point of the expansion algorithm. A call to
     * this method triggers the expansion algorithm to run, generating cumulative
     * costs and parents pointers.
     *
     * @param seed the starting seedpoint
     */
    public void addSeed(CvPoint seed) {
        addSeed(seed.y(), seed.x());
    }

    /**
     * Sets a seedpoint as the starting point of the expansion algorithm. A call to
     * this method triggers the expansion algorithm to run, generating cumulative
     * costs and parents pointers.
     *
     * @param row the pixel row of the starting seedpoint
     * @param col the column of the starting seedpoint
     */
    public void addSeed(int row, int col) {
        System.out.println("New seed-point row:" + row + " col:" + col);
        lastSeed = costs[row][col];
        reset();
        expand(row, col);
    }

    /**
     * @return the Node corresponding to the specified point in the image
     */
    public Node getNode(CvPoint point) {
        return getNode(point.y(), point.x());
    }

    /**
     * @return the Node corresponding to the specified point in the image
     */
    public Node getNode(int row, int col) {
        return costs[row][col];
    }

    public Node snapToEdge(int row, int col, int dist){
        Node current = original[row][col];
        Node best = current;

        for (int i = row - dist; i < row + dist; i++) {
            for (int j = col - dist; j < col + dist; j++) {
                current = original[i][j];
                if (current.cost < best.cost)
                    best = current;
            }
        }

        return getNode(best.row, best.col);
    }

    public Node getClosestEdge(int row, int col){
        Node n = costs[row][col];
        int count = 0;
        while (!n.equals(lastSeed) && n.parent != null){
            if (original[n.row][n.col].cost <= 0)
                return n;
            if (++count >= 30) break; //if too far away
            n = n.parent;
        }

        n = costs[row][col];
        Node best = n;
        ArrayList<Node> neighbors = getNeighbors(n, 6);
        while (!n.equals(lastSeed) && n.parent != null){
            if (neighbors.contains(n) &&
                    original[n.row][n.col].cost < original[best.row][best.col].cost){
                best = n;
            }
            if (original[best.row][best.col].cost <= 5) return best;

            n = n.parent;
        }

        return best;
    }

    /**
     * Generates cumulative costs and parent pointers using a variation of Dijkstra's
     * shortest path algorithm. The resultant min-cost tree is stored in a 2-D array
     */
    private void expand(int row, int col) {
        //create algorithm data structures COMPARE VIA CUM COST
        Set<Node> closed = new HashSet<Node>();
        PriorityQueue<Node> wavefront =
                new PriorityQueue<Node>(2000, new Comparator<Node>()
                {
                    @Override
                    public int compare(Node a, Node b) {
                        if (a.cost > b.cost) return 1;
                        if (a.cost < b.cost) return -1;
                        return 0;
                    }
                });

        Node current = costs[row][col];
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
            List<Node> neigbors = getNeighbors(current, 1);
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

    /**
     * @return a list of n's neighbor Nodes
     */
    private ArrayList<Node> getNeighbors(Node n, int distance) {
        ArrayList<Node> neighbors = new ArrayList<Node>(8);

        for (int i = n.row - distance; i <= n.row + distance; i++) {
            for (int j = n.col - distance; j <= n.col + distance; j++) {
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

    /**
     * @return the euclidean-scaled cumulative cost from Node current to Node n
     */
    private int euclideanAdd(Node current, Node n) {
        //if diagonal, scale by RAD2
        if (current.row != n.row && current.col != n.col)
            return current.cost + (int) (RAD2 * costs[n.row][n.col].cost);

        return current.cost + costs[n.row][n.col].cost;
    }

    /**
     * @return the euclidean distance between Node a and Node b
     */
    private double euclideanDist(Node a, Node b) {
        return (double) Math.sqrt(((b.row - a.row) * (b.row - a.row)) +
                ((b.col - a.col) * (b.col - a.col)));
    }

}
