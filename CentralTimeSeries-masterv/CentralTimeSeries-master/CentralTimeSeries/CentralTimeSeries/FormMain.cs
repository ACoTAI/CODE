using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;

namespace CentralTimeSeries
{
    public partial class FormMain : Form
    {
        private TimeSeries test1, test2;
        public FormMain()
        {
            InitializeComponent();
            initDataset();
            test1 = new TimeSeries();
            test2 = new TimeSeries();
            test1.length = 5;
            test2.length = 5;
            double[] v1 = { 1, 2, 3, 4, 5 };
            double[] v2 = { 3, 4, 5, 6, 7 };
            //test1.length = 2;
            //test2.length = 3;
            //double[] v1 = { 1, 2 };
            //double[] v2 = { 3, 4, 5 };
            test1.data = v1;
            test2.data = v2;
        }
        private class Matching
        /*
            The class corresponds to an element of the matrix used in 
            g(dp)^2 algorithm.
            Assuming the subscript is (i,j), and c(1:k) is the central
            time series of x(1:i) and y(1:j).
         */ 
        {
            public double v, e;
            //v is the value of c_k, e is WGSS(c,{x,y})
            public int pi, pj; 
            //(i,j)-th problem is convered into (pi,pj)-th subprolem.
            public int ia, ib, ja, jb;
            /* c_k matches to (x_{ia},...,x_{ib}) of x, 
               and matches to (y_{ja},...,y_{jb}) of y
             */
            public int length;
            /*
             The length of the central time series
            */ 
            public Matching()
            {
                e = 1.0e20;
                v = 0.0;
                pi = pj = 0;
                ia = ib = ja = jb = 0;
                length = 0;
            }
        };

        private class TimeSeries
        {
            public int index;
            public int classIndex;
            public double[] data;
            public double weight;
            public int length;
            public TimeSeries()
            {
                weight = 1.0;
                index  = classIndex = -1;
            }
            public void parseData(int index, string dataLine)
            {
                try
                {
                    this.index = index;
                    string[] strList = dataLine.Split(
                        new char[] { ' ' },
                        StringSplitOptions.RemoveEmptyEntries
                    );
                    classIndex = (int)(double.Parse(strList[0]) + 0.1);
                    length = strList.Length - 1;
                    data = new double[length];

                    double mean = 0.0;
                    for (int i = 0; i < length; i++)
                    {
                        data[i] = double.Parse(strList[i + 1]);
                        mean += data[i];
                    }
                    mean /= length;
                    double delta = 0.0;
                    for (int i = 0; i < length; i++)
                    {
                        delta += (data[i] - mean) * (data[i] - mean);
                    }
                    delta = Math.Sqrt(delta/length);
                    for (int i = 0; i < length; i++)
                    {
                        data[i] = (data[i] - mean) / delta;
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.ToString(), "");
                }
            }
            public void copy(TimeSeries ts)
            {
                this.length = ts.length;
                this.data = new double[ts.length];
                for (int i = 0; i < ts.length; i++)
                    this.data[i] = ts.data[i];
                this.classIndex = ts.classIndex;
            }
            public TimeSeries clone()
            {
                TimeSeries ts = new TimeSeries();
                ts.length = this.length;
                ts.data = new double[ts.length];
                for (int i = 0; i < ts.length; i++)
                    ts.data[i] = this.data[i];
                ts.classIndex = this.classIndex;
                return ts;
            }
        }
        static double MAX_VALUE = 1.0e20;

        private class DTWNode
        /*
         The class corresponds to an element of a given DTW matrix
        */
        {
            public int pi, pj; 
            /*
              If the current DTWNode is a node of a DTW path, 
              (pi,pj) is the predecessor subscript in the path.
            */
            public double error; 
            /*
              If the subscript of DTWNode in a DTW matrix is (i,j),
              error is DTW(x(1:i),y(1:j))
            */
            public DTWNode()
            {
                pi = pj = -1;
                error = MAX_VALUE;
            }
        }
        private class TimeSeriesPair
        {   
            /* 
             The class includes DTW, NLAAF, SDTW, DBA and gDPDP(g(dp)^2) 
             algorithms for two given time series
            */
            public TimeSeries xs, ys;
            public TimeSeriesPair(TimeSeries ts1, TimeSeries ts2)
            {
                xs = ts1;
                ys = ts2;
            }

            private DTWNode[][] matrix_DTW; //DTW matrix
            private DTWNode[] path_DTW; //DTW path
            public int[][] associates;
            /* Let a = associates[i][0], and let b = associates[i][1], 
             y_a, y_{a+1},..., y_b are all the entries matching to x_i
             in a DTW path.
             */ 
            public double error = MAX_VALUE; // the result of DTW distance
            public static int PATH_OFFSET = 100;
            void updateMatrix_DTW(int i, int j, int pi, int pj, double dist)
                //update DTW matrix
            {
                double error;
                if ((pi == -1) && (pj == -1))
                    error = dist;
                else if ((pi == -1) || (pj == -1))
                    error = 1.0e20 + dist;
                else
                    error = matrix_DTW[pi][pj].error + dist;
                if (error < matrix_DTW[i][j].error)
                {
                    matrix_DTW[i][j].error = error;
                    matrix_DTW[i][j].pi = pi;
                    matrix_DTW[i][j].pj = pj;
                }
            }
            public void DTW()
            {
                int m = xs.length;
                int n = ys.length;
                associates = new int[m][];
                matrix_DTW  = new DTWNode[m][];
                for (int i = 0; i < m; i++)
                {
                    matrix_DTW[i] = new DTWNode[n];
                    for (int j = 0; j < n; j++)
                    {
                        matrix_DTW[i][j] = new DTWNode();
                        matrix_DTW[i][j].error = 1.0e20;
                    }
                    associates[i] = new int[2];
                }
                
                for (int i = 0; i < m; i++)
                {
                    //int ja = Math.Max(0, i * n / m - PATH_OFFSET); //Math.Max(0, i * n / m - n / 3);
                    //int jb = Math.Min(n, i * n / m + PATH_OFFSET); //Math.Min(n, i * n / m + n / 3);
                    //int ja = Math.Max(0, i * n / m - 2 * n / 3);
                    //int jb = Math.Min(n, i * n / m + 2 * n / 3);
                    int ja = 0;
                    int jb = n;
                    for (int j = ja; j < jb; j++)
                    {
                        double dist = Math.Abs(xs.data[i] - ys.data[j]);
                        dist = dist * dist;
                        updateMatrix_DTW(i, j, i - 1, j - 1, dist);
                        updateMatrix_DTW(i, j, i - 1, j, dist);
                        updateMatrix_DTW(i, j, i, j - 1, dist);
                        //from the DTW recursive relationship
                    }
                }
                this.error = matrix_DTW[m - 1][n - 1].error;
                DTWNode node = new DTWNode();
                node.pi = m - 1;
                node.pj = n - 1;
                node.error = this.error;
                Stack nodeStack = new Stack();
                nodeStack.Push(node);
                while (true) //retrieve DTW path from matrix_DTW to nodeStack
                {
                    node = matrix_DTW[node.pi][node.pj];
                    if ((node.pi == -1) || (node.pj == -1))
                        break;
                    nodeStack.Push(node);
                }
                path_DTW = new DTWNode[nodeStack.Count];
                for (int i = 0;  i < path_DTW.Length; i ++)
                {
                    path_DTW[i] = (DTWNode)nodeStack.Pop();
                }
                matrix_DTW = null;
                int pa = 0, pb = 0, p = 0;
                for (int i = 0; i < m; i++)
                {
                    for (; p < path_DTW.Length && path_DTW[p].pi == i; p++) ;
                    pb = p - 1;
                    associates[i][0] = path_DTW[pa].pj;
                    associates[i][1] = path_DTW[pb].pj;
                    //calculate the matchings of x_i

                    pa = p;
                }
            }
            public Matching[][] matrix_gDPDP;
            double avg(TimeSeries xs, int ia, int ib, TimeSeries ys, int ja, int jb)
            {
                //calculate the average of x_{ia},...,x_{ib} and y_{ja},...,y_{jb}
                double cnt = xs.weight * (ib - ia + 1) + ys.weight * (jb - ja + 1);
                if (Math.Abs(cnt) < 0.000001) return 0.0;
                double sum = 0;
                for (int i = ia; i <= ib; i++)
                    sum += xs.weight * xs.data[i];
                for (int j = ja; j <= jb; j++)
                    sum += ys.weight * ys.data[j];
                return (sum / cnt);
            }
            double err(TimeSeries xs, int ia, int ib, TimeSeries ys, int ja, int jb)
            {
                //calculate the mean squared error of x_{ia},...,x_{ib} and y_{ja},...,y_{jb}
                double temp = avg(xs, ia, ib, ys, ja, jb);
                double sum = 0;
                for (int i = ia; i <= ib; i++)
                    sum += xs.weight * (xs.data[i] - temp) * (xs.data[i] - temp);
                for (int j = ja; j <= jb; j++)
                    sum += ys.weight * (ys.data[j] - temp) * (ys.data[j] - temp);
                return sum;
            }
            void updateMatrix_gDPDP(int i, int j, int pi, int pj, int ia, int ja, TimeSeries xs, TimeSeries ys)
            {
                //update the matrix of g(dp)^2 algorithm
                int prevCenterLength = 0;
                double tmpError = err(xs, ia, i, ys, ja, j);
                if ((pi == -1) && (pj == -1))
                    tmpError += 0.0;
                else if ((pi == -1) || (pj == -1))
                    tmpError += 1.0e20;
                else
                {
                    tmpError += matrix_gDPDP[pi][pj].e;
                    prevCenterLength = matrix_gDPDP[pi][pj].length;
                }
                Matching pm = matrix_gDPDP[i][j];
                if (tmpError < pm.e)
                //Assure that pi,pj,ia,ib,ja,jb are the argmin...
                {
                    pm.e = tmpError;
                    pm.pi = pi;
                    pm.pj = pj;
                    pm.ia = ia;
                    pm.ib = i;
                    pm.ja = ja;
                    pm.jb = j;
                    pm.length = prevCenterLength + 1;
                }
            }
            public static int BAND_SIZE = 100;
            public TimeSeries gDPDP()
            {
                matrix_gDPDP = new Matching[xs.length][];
                for (int i = 0; i < xs.length; i++)
                {
                    matrix_gDPDP[i] = new Matching[ys.length];
                    for (int j = 0; j < ys.length; j++)
                    {
                        matrix_gDPDP[i][j] = new Matching();
                    }
                }
                int nx = xs.length;
                int ny = ys.length;
                BAND_SIZE = TimeSeriesPair.PATH_OFFSET;
                for (int i = 0; i < nx; i++)
                {
                    //int ja = Math.Max(0, (i * ny / nx) - 2 * BAND_SIZE);
                    //int jb = Math.Min(ny, (i * ny / nx) + 2 * BAND_SIZE);
                    int ja = 0;
                    int jb = ny;
                    for (int j = ja; j < jb; j++)
                    {
                        for (int p = i; p >= Math.Max(0, i - BAND_SIZE); p--)
                        //for (int p = i; p >= 0; p--)
                        {
                            //from the recursive relationship of g(dp)^2
                            updateMatrix_gDPDP(i, j, p - 1, j - 1, p, j, xs, ys);
                            updateMatrix_gDPDP(i, j, p - 1, j, p, j, xs, ys);
                        }
                        for (int q = j; q >= Math.Max(0, j - BAND_SIZE); q--)
                        //for (int q = j; q >= 0; q--)
                        {
                            //from the recursive relationship of g(dp)^2
                            updateMatrix_gDPDP(i, j, i - 1, q - 1, i, q, xs, ys);
                            updateMatrix_gDPDP(i, j, i, q - 1, i, q, xs, ys);
                        }
                    }
                }

                double resError = matrix_gDPDP[nx - 1][ny - 1].e;
                //WGSS 
                int centerLen = matrix_gDPDP[nx - 1][ny - 1].length;
                //the length of central time series

                TimeSeries center = new TimeSeries();
                center.length = centerLen;
                center.data = new double[centerLen];

                int ki = nx - 1, kj = ny - 1;
                while ((ki > -1) && (kj > -1))
                //retrieve central time series from matrix_gDPDP to matStack
                {
                    Matching tmpMat = matrix_gDPDP[ki][kj];
                    double ck = avg(xs, tmpMat.ia, tmpMat.ib, ys, tmpMat.ja, tmpMat.jb);
                    center.data[--centerLen] = ck;
                    ki = tmpMat.pi;
                    kj = tmpMat.pj;
                }
                center.weight = xs.weight + ys.weight;
                return center;
            }
        }

        private class DataSet
        {
	        public string dataName;
            public int classCount;
            public int dataCount;
            public int dataSize;
            public ArrayList timeSeriesList;
            public DataSet(string dataName, int classCount, int dataSize)
            {
                this.dataName = dataName;
                this.classCount = classCount;
                this.dataSize = dataSize;
                timeSeriesList = new ArrayList();
                List<string> listLines = new List<string>();
                string fileName = 
                    System.Windows.Forms.Application.StartupPath + "\\dataset1\\" +
                    dataName + "\\" + dataName + "_TRAIN";
                using (StreamReader reader = new StreamReader(fileName))
                {
                    string line = reader.ReadLine();
                    while (line != "" && line != null)
                    {
                        listLines.Add(line);
                        line = reader.ReadLine();
                    }
                }
                fileName =
                    System.Windows.Forms.Application.StartupPath + "\\dataset1\\" +
                    dataName + "\\" + dataName + "_TEST";
                using (StreamReader reader = new StreamReader(fileName))
                {
                    string line = reader.ReadLine();
                    while (line != "" && line != null)
                    {
                        listLines.Add(line);
                        line = reader.ReadLine();
                    }
                }
                this.dataCount = listLines.Count;
                for (int i = 0; i < listLines.Count; i++)
                {
                    TimeSeries ts = new TimeSeries();
                    ts.parseData(i, listLines[i]);
                    timeSeriesList.Add(ts);
                }
            }
            private class DTWPair
            {
                public double dtw;
                public int i0, i1;
                public double w0, w1;
                public DTWPair(double dtw, int i0, double w0, int i1, double w1)
                {
                    this.dtw = dtw;
                    this.i0 = i0;
                    this.w0 = w0;
                    this.i1 = i1;
                    this.w1 = w1;
                }
                public DTWPair clone()
                {
                    return new DTWPair(dtw, i0, w0, i1, w1);
                }
            }
            private DTWPair getPairFromList(ArrayList pairList)
            {
                Hashtable pairHash = new Hashtable();
                for (int i = 0; i < pairList.Count; i++)
                {
                    DTWPair pair = (DTWPair)pairList[i];
                    if (pairHash.Contains(pair.i0))
                    {
                        DTWPair temp = (DTWPair)pairHash[pair.i0];
                        if (pair.dtw < temp.dtw)
                        {
                            temp.i1 = pair.i1;
                            temp.dtw = pair.dtw;
                        }
                    }
                    else
                    {
                        pairHash[pair.i0] = pair.clone();
                    }
                }
                DTWPair res = null;
                double maxValue = 0.0;
                foreach (DictionaryEntry de in pairHash)
                {
                    DTWPair pair = (DTWPair)de.Value;
                    if (pair.dtw > maxValue)
                    {
                        maxValue = pair.dtw;
                        res = pair.clone();
                    }
                }
                ArrayList samples = new ArrayList();
                for (int i = 0; i < pairList.Count; i++)
                {
                    DTWPair pair = (DTWPair)pairList[i];
                    if ((pair.i0 == res.i0) || (pair.i0 == res.i1) ||
                        (pair.i1 == res.i0) || (pair.i1 == res.i1))
                        continue;
                    samples.Add(pair);
                }
                pairList.Clear();
                for (int i = 0; i < samples.Count; i++)
                {
                    pairList.Add(samples[i]);
                }
                return res;
            }
            private DTWPair[] getPairsFromList(ArrayList pairList, int size)
            {
                DTWPair[] pairs = new DTWPair[size];
                for (int i = 0; i < size; i++)
                    pairs[i] = getPairFromList(pairList);
                return pairs;
            }
            public double mDPDP(
                ArrayList samples, 
                ListBox lbx, int classIndex, int classCount, 
                TimeSeries seed)
            {
                if (samples.Count == 0) return 0.0;
                double error = 0.0;
                int groupCount = 4;
                TimeSeries[] seedList = new TimeSeries[groupCount];
                int[] numberList = new int[samples.Count];
                ArrayList[] dataList = new ArrayList[groupCount];
                for (int i = 0; i < samples.Count; i++)
                    numberList[i] = i;
                Random random = new Random();
                for (int i = 0; i < groupCount; i++)
                {
                    dataList[i] = new ArrayList();
                    int lastIndex = samples.Count - i - 1;
                    int randIndex = random.Next(lastIndex + 1);
                    int randValue = numberList[randIndex];
                    numberList[randIndex] = numberList[lastIndex];
                    seedList[i] = (TimeSeries)samples[randValue];
                }
                int tempGroup = groupCount;
                while (tempGroup > 0)
                {
                    for (int i = 0; i < tempGroup; i++)
                    {
                        dataList[i].Clear();
                        seedList[i].weight = 1.0;
                    }
                    double[][] dtwMatrix = new double[samples.Count][];
                    for (int i = 0; i < samples.Count; i++)
                    {
                        TimeSeries ts = (TimeSeries)samples[i];
                        dtwMatrix[i] = new double[tempGroup];
                        for (int j = 0; j < tempGroup; j++)
                        {
                            TimeSeriesPair pair = new TimeSeriesPair(seedList[j], ts);
                            pair.DTW();
                            dtwMatrix[i][j] = pair.error;
                        }
                    }

                    for (int i = 0; i < samples.Count; i++)
                    {
                        TimeSeries ts = (TimeSeries)samples[i];
                        int k = 0;
                        double minError = 1.0e20;
                        for (int j = 0; j < tempGroup; j++)
                        {
                            if (dataList[j].Count > (samples.Count / tempGroup + 1))
                                continue;
                            if (dtwMatrix[i][j] < minError)
                            {
                                minError = dtwMatrix[i][j];
                                k = j;
                            }
                        }
                        dataList[k].Add(ts);
                    }

                    for (int i = 0; i < tempGroup; i++)
                    {
                        int loopMax = (tempGroup == 1) ? 30 : 15;
                        for (int j = 0; j < loopMax; j++)
                        {
                            error = dbaStep(seedList[i], dataList[i]);
                            string line = string.Format(
                                    "class = {0:d}/{1:d}; size = {2:d}; group = {3:d}/{4:d}; loop = {5:d}; error = {6:f}, avgError = {7:f}",
                                    classIndex, classCount, dataList[i].Count, i + 1, tempGroup, j, error, error / dataList[i].Count
                                );
                            lbx.Items.Add(line);
                            lbx.SetSelected(lbx.Items.Count - 1, true);
                            Application.DoEvents();
                        }
                    }
                    if (tempGroup > 1)
                    {
                        ArrayList pairList = new ArrayList();
                        //calculate the pairwise DTW distances among the group seeds
                        for (int i = 0; i < tempGroup; i++)
                        {
                            for (int j = 0; j < tempGroup; j++)
                            {
                                if (i == j) continue;
                                TimeSeriesPair pair = new TimeSeriesPair(seedList[i], seedList[j]);
                                pair.DTW();
                                pairList.Add(new DTWPair(pair.error,
                                    i, dataList[i].Count,
                                    j, dataList[j].Count
                                ));
                            }
                        }
                        DTWPair[] pairVect = getPairsFromList(pairList, tempGroup / 2);
                        //Each element of pairVect is a pair of indice which corresponding seeds will be merged.

                        TimeSeries[] newSeeds = new TimeSeries[tempGroup / 2];
                        for (int i = 0; i < tempGroup / 2; i++)
                        {
                            int i0 = pairVect[i].i0;
                            int i1 = pairVect[i].i1;
                            double w0 = pairVect[i].w0;
                            double w1 = pairVect[i].w1;
                            TimeSeries ts0 = (TimeSeries)seedList[i0];
                            TimeSeries ts1 = (TimeSeries)seedList[i1];
                            if (Math.Abs(w0 + w1) < 0.000001)
                            {
                                newSeeds[i] = ts0.clone();
                            }
                            else
                            {
                                ts0.weight = w0 / (w0 + w1);
                                ts1.weight = w1 / (w0 + w1);
                                TimeSeriesPair pair = new TimeSeriesPair(ts0, ts1);
                                newSeeds[i] = pair.gDPDP();
                                //Merge the two old seeds into a new seed.
                            }
                        }
                        for (int i = 0; i < tempGroup / 2; i++)
                        {
                            seedList[i] = newSeeds[i];
                            //update new seeds
                        }
                    }
                    tempGroup /= 2;
                    //the group count becomes the half

                }
                seed.copy(seedList[0]);
                return error;
            }
            public double dbaStep(TimeSeries seed, ArrayList samples)
            {
                TimeSeriesPair[] results = new TimeSeriesPair[samples.Count];
                double error = 0.0;
                for (int i = 0; i < samples.Count; i++)
                {
                    TimeSeries ts = (TimeSeries)samples[i];
                    results[i] = new TimeSeriesPair(seed, (TimeSeries)samples[i]);
                    results[i].DTW();
                    error += results[i].error;
                }

                int m = seed.length;
                for (int i = 0; i < m; i++)
                {
                    double mean = 0.0;
                    int n = 0;
                    for (int j = 0; j < samples.Count; j++)
                    {
                        TimeSeries ts = (TimeSeries)samples[j];
                        int k0 = results[j].associates[i][0];
                        int k1 = results[j].associates[i][1];
                        for (int k = k0; k <= k1; k++)
                        {
                            mean += ts.data[k];
                            n++;
                        }
                    }
                    seed.data[i] = mean / n;
                }

                return error;
            }
            public double DBA( int classIndex ) {
                ArrayList samples = new ArrayList();
                for (int i = 0; i < dataCount; i++)
                {
                    TimeSeries ts = (TimeSeries)timeSeriesList[i];
                    if ((classIndex == -1) || (classIndex == ts.classIndex))
                        samples.Add(ts);
                }
                if (samples.Count == 0) return 0.0;
                Random random = new Random();
                int index = random.Next(samples.Count);
                TimeSeries seed = ((TimeSeries)samples[index]).clone();

                double error = 0.0;
                for (int i = 0; i < 30; i++)
                {
                    error = dbaStep(seed, samples);
                }
                return error;
            }


        }
        DataSet[] datasetList={
	        new DataSet("50words",50,270),//0
	        new DataSet("Adiac", 37,176),//1
	        new DataSet("Beef", 5, 470),//2
	        new DataSet("CBF", 3, 128),//3
	        new DataSet("Coffee", 2, 286),//4
	        new DataSet("ECG200", 2, 96),//5
	        new DataSet("FaceAll", 14, 131),//6
	        new DataSet("FaceFour", 4, 350),//7
	        new DataSet("fish", 7, 463),//8
	        new DataSet("Gun_Point", 2, 150),//9
	        new DataSet("Lighting2", 2, 637),//10
	        new DataSet("Lighting7", 7, 319),//11
	        new DataSet("OliveOil", 4, 570),//12
	        new DataSet("OSULeaf", 6, 427),//13
	        new DataSet("SwedishLeaf", 15,128),//14 
	        new DataSet("synthetic_control", 6, 60),//15
	        new DataSet("Trace", 4, 275),//16
	        new DataSet("Two_Patterns", 4, 128),//17
	        new DataSet("wafer", 2, 152),//18
	        new DataSet("yoga", 2, 426)//19
        };
        Hashtable datasetHash = new Hashtable();
        private void initDataset()
        {
            for (int i = 0; i < datasetList.Length; i++)
            {
                datasetHash[datasetList[i].dataName] = datasetList[i];
            }
        }

        private void btnClear_Click(object sender, EventArgs e)
        {
            lbxData.Items.Clear();
        }

        private void showMessage(TimeSeriesPair res, TimeSeries center)
        {

            TimeSeriesPair m13 = new TimeSeriesPair(test1, center);
            m13.DTW();
            TimeSeriesPair m23 = new TimeSeriesPair(test2, center);
            m23.DTW();
            string line = "error = " + string.Format("{0:f}", m13.error + m23.error);
            lbxData.Items.Add(line);
        }

        private void btn_gDPDP_Click(object sender, EventArgs e)
        {
            TimeSeriesPair res = new TimeSeriesPair(test1, test2);
            res.DTW();
            TimeSeries center = res.gDPDP();
            showMessage(res, center);
        }

        private void btn_DBA2_Click(object sender, EventArgs e)
        {
        }

        private void btn_SDTW_Click(object sender, EventArgs e)
        {
        }

        private void btn_mDPDP_Click(object sender, EventArgs e)
        {
            DataSet ds = (DataSet)datasetHash[cmbDataName.Text];
            TimeSeries seed = new TimeSeries();
            double sum = 0.0;
            int n = 0;
            for (int i = 0; i <= ds.classCount; i++)
            {
                lbxData.Items.Add("");
                lbxData.Items.Add(string.Format("---------- classIndex = {0:d} ----------", i));
                ArrayList samples = new ArrayList();
                for (int j = 0; j < ds.dataCount; j++)
                {
                    TimeSeries ts = (TimeSeries)ds.timeSeriesList[j];
                    if ((i == -1) || (i == ts.classIndex))
                        samples.Add(ts);
                }
                double minError = 1.0e20;
                int tryNumber = 3;
                for (int j = 0; j < tryNumber; j++)
                {
                    lbxData.Items.Add(string.Format("---------- try number = {0:d} ----------", j));
                    double tempError = ds.mDPDP(samples, lbxData, i, ds.classCount, seed);
                    //double tempError = ds.mDPDP(samples, seed);
                    if (tempError < minError)
                        minError = tempError;
                }
                sum += minError;
                n += samples.Count;
                lbxData.Items.Add(string.Format("index = {0:d}, average = {1:f}", i, sum/n));
                lbxData.Items.Add("---------------------------");
                lbxData.Items.Add("");
            }
            sum /= ds.dataCount;
            lbxData.Items.Add(string.Format("average = {0:f6}", sum));
        }

        private void cmbDataName_SelectedIndexChanged(object sender, EventArgs e)
        {
            DataSet ds = (DataSet)datasetHash[cmbDataName.Text];
            test1 = (TimeSeries)ds.timeSeriesList[0];
            test2 = (TimeSeries)ds.timeSeriesList[1];
        }

    }
}
