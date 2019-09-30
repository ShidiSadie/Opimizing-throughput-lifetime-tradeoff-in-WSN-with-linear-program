#include <ilcplex/ilocplex.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <Eigen/Dense>  
#include <malloc.h>
#include <time.h>
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

using namespace std;
ILOSTLBEGIN
#define q 2
typedef IloArray<IloNumVarArray> NumVarMatrix;
typedef IloArray<IloNumArray>    NumMatrix;

//设置常数
const float P[] = { 0.0032, 0.032, 0.1, 0.2, 0.32, 0.512, 0.8192, 1.0 };
const int plength = 8;
//const IloNumArray p(env, 8, 8.5, 9.9, 11.2, 12.5, 13.9, 15.2, 16.5, 17.4);
const float N0 = 0.0001;
//const IloNum U = 1.8;
//const IloNum TI = 19.7;
//const IloNum Emax = 0.0006264;
//const IloNum Ie = 1;
const float dp = 1000;
const float R[] = { 250, 500, 1000, 2000 };
const int rlength = 4;
const float Th[] = { 2, 4, 8, 16 };
const float Rmax = 100;
const float c = 300;
int em=1;
int cc = 0;

typedef struct P{
	float x;
	float y;
}point;

typedef struct L{
	int s;
	int e;
	int num;
}link;

typedef struct V{
	int s;
	int e;
	int r;
	int num;
}vertice;

typedef struct bm{
	int s;
	int r;
	float p;
}bimatch;

typedef struct SS{
	float dist;
	int num;
}trans;

float distance(point *p1, point *p2, int i, int j)
{
	return((p1[i].x - p2[j].x)*(p1[i].x - p2[j].x) + (p1[i].y - p2[j].y)*(p1[i].y - p2[j].y));
}

int sched(int *m, int count, vertice *v, trans **SS)
{
	MatrixXf A(count-1,count-1);
	MatrixXf A2(count - 1, count - 1);
	MatrixXf B(count - 1, count - 1);
	MatrixXf C(count - 1, 1);
	MatrixXf P1(count - 1, 1);
	MatrixXf I(count - 1, count - 1);
	int i,j;
	float f;
	if (count == 2)
	{
		if (SS[v[m[1]].s][v[m[1]].e].dist <= P[plength - 1] / (Th[v[m[1]].r] * N0))
			return 1;
		else return 0;
	}
	for (i = 0; i <= count - 2; i++)
	{
		for (j = 0; j <= count - 2; j++)
		{
			if (i == j)
			{
				A(i, j) = 0;
				B(i, j) = Th[v[m[i+1]].r];
				I(i, j) = 1;
			}
			else
			{
				A(i, j) = SS[v[m[i+1]].s][v[m[i+1]].e].dist / SS[v[m[j+1]].s][v[m[i+1]].e].dist;
				B(i, j) = 0;
				I(i, j) = 0;
			}
		}
	}
	for (i = 0; i <= count - 2; i++)
	{
		C(i,0) = N0*SS[v[m[i+1]].s][v[m[i+1]].e].dist;
	}
	A2 = B*A;
	EigenSolver<MatrixXf> es(A2);
	MatrixXcf evals = es.eigenvalues();//获取矩阵特征值 4*1
	MatrixXf evalsReal;//注意这里定义的MatrixXd里没有c
	evalsReal = evals.real();//获取特征值实数部分
	f = evalsReal.rowwise().sum().maxCoeff();
	if (f < 1)
	{
		int t = 0;
		P1 = ((I - B*A).inverse())*(B*C);
		for (i = 0; i <= count - 2; i++)
		{
			if (P1(i, 0)>P[plength-1])
			{
				t = 1;
				break;
			}
		}
		if (t == 0)
		{
			return 1;
		}
		else return 0;
	}
	else return 0; 
}

int conf(int i, int j, vertice *v, trans **SS)
{
	float con1, con2, con3, bi, bj, aij, aji, ci, cj;
	bi = Th[v[i].r];
	bj = Th[v[j].r]; 
	aij = SS[v[i].s][v[i].e].dist/SS[v[j].s][v[i].e].dist;
	aji = SS[v[j].s][v[j].e].dist / SS[v[i].s][v[j].e].dist;
	ci = N0*SS[v[i].s][v[i].e].dist;
	cj = N0*SS[v[j].s][v[j].e].dist;
	con1 = bi*bj*aij*aji;
	con2 = bi*(ci+bj*aij*cj)/(1-bi*bj*aij*aji);
	con3 = bj*(cj + bi*aji*ci) / (1 - bi*bj*aij*aji);
	if (con1 < 1 && con2 <= P[plength - 1] && con3 <= P[plength - 1])
		return 0;
	else return 1;
}

void quick_sort(vertice *v, double *w, int l, int r)
{
	if (l < r)
	{
		int i = l, j = r,e=v[l].e,s=v[l].s,num=v[l].num,r2=v[l].r;
		double ww = w[l];
		double x = R[v[l].r] * w[l];
		while (i < j)
		{
			while (i < j && R[v[j].r] * w[j] <= x) // 从右向左找第一个小于x的数  
				j--;
			if (i < j)
			{
				v[i].s = v[j].s;
				v[i].e = v[j].e;
				v[i].num = v[j].num;
				v[i].r = v[j].r;
				w[i] = w[j];
				i++;
			}

			while (i < j && R[v[i].r] * w[i] > x) // 从左向右找第一个大于等于x的数  
				i++;
			if (i < j)
			{
				v[j].s = v[i].s;
				v[j].e = v[i].e;
				v[j].num = v[i].num;
				v[j].r = v[i].r;
				w[j] = w[i];
				j--;
			}
		}
		v[i].e = e;
		v[i].s = s;
		v[i].num = num;
		v[i].r = r2;
		w[i] = ww;
		quick_sort(v, w, l , i - 1); // 递归调用   
		quick_sort(v, w, i + 1, r);
	}
}

int mgeneration(vertice *v,int vcount,double *w,int *taken,bimatch **b,int mcount,trans **SS,int count2)
{
	int **plcg = new int*[vcount];
	double weight = 0;
	int head = 1;
	int *m = new int[vcount];
	int i, j, k = 0,countm=1;
	int all=0;
	int *visited = new int[vcount];
	for (i = 1; i <= vcount - 1; i++)
		visited[i] = 0;
	//按照w*r排序
	quick_sort(v,w,1,vcount-1);
	//构建PLCG图
	if (mcount > 1)
	{
		for (i = 1; i <= count2 - 1; i++)
			all += b[mcount - 1][i].s;
	}
	else all = 0;
	/*for (i = 1; i <= vcount - 1; i++)
	{
		cout << v[i].s << "->" << v[i].e << ":" << "r=" << v[i].r << " " << "num=" << v[i].num << "weight=" << w[i];
		cout << endl;
	}*/
	for (j = 1; j<=vcount-1; j++)
	{
		plcg[j] = new int[vcount - 1];
	}
	for (i = 1; i <= vcount-1; i++)
	for (j = 1; j <= vcount-1; j++)
	{
		plcg[i][j] = 0;
	}
	for (i = 1; i <= vcount-1; i++)
	for (j = i + 1; j <= vcount - 1; j++)
	{
		if (v[i].num == v[j].num || v[i].e == v[j].e || v[i].e == v[j].s || v[i].s == v[j].e || v[i].s == v[j].s || conf(i, j, v, SS))
		{
			plcg[i][j] = 1;
			plcg[j][i] = 1;
		}
	}
	while (head <= vcount - 1)
	{
		
		for (i = head + k; i <= vcount - 1; i++)
		{
			int flag = 0;
			if (!visited[i] && !taken[i])
			for (j = 1; j <= countm - 1; j++)
			{
				if (plcg[i][m[j]])
				{
					flag = 1;
					break;
				}
			}
			if (!visited[i] && !taken[i] && flag == 0)
			{
				m[countm] = i;
				countm++;
				weight = weight + w[i] * R[v[i].r];
			}
		}
		int flag2 = 0;
		int flag3 = 0;
		int flag4 = 0;
		if (weight > 1)
		{
			/*for (i = 1; i <= countm - 1; i++)
			{
				cout << v[m[i]].s << "->" << v[m[i]].e << ":" << "r=" << v[m[i]].r << " " << "num=" << v[m[i]].num << "weight=" << w[m[i]];
			}
			cout << endl;*/
			if (mcount >= 2)
			{
				for (i = 1; i <= countm - 1; i++)
				{
					if (b[mcount - 1][v[m[i]].num].s && b[mcount - 1][v[m[i]].num].r == v[m[i]].r)
					{
						flag3 = 1;
					}
					else
					{
						flag3 = 0;
						break;
					}
				}
			}
			if (flag3 && all == countm - 1) flag4 = 1; else flag4 = 0;
			if (flag4) return 0;
			if (sched(m, countm, v,SS))
			{
				//cout << weight<<endl;
				/*for (i = 1; i <= countm - 1; i++)
				{
					cout << v[m[i]].s << "->" << v[m[i]].e << ":" << "r=" << v[m[i]].r << " " << "num=" << v[m[i]].num << "weight=" << w[m[i]];
					cout << endl;
				}*/
				for (i = 1; i <= countm - 1; i++)
				{
					b[mcount][v[m[i]].num].s = 1;
					b[mcount][v[m[i]].num].r = v[m[i]].r;
					taken[m[i]] = 1;
				}
				return 1;
			}
			else
			{
				flag2=1;
			}
		}
		if (weight <= 1 || flag2)
		{
			if (countm > 2)
			{
				k = m[countm-1];
				visited[m[countm - 1]] = 1;
				weight -= w[m[countm - 1]] * R[v[m[countm - 1]].r];
				countm--;
			}
			else
			{
				head++;
				countm = 1;
				for (i = 1; i <= vcount - 1; i++)
					visited[i] = 0;
				weight = 0;
				k = 0;
			}
		}
		
	}
	return 0;
	
}

void domixlinear(IloEnv env, int **S_T, int **S_L, int **E_L, bimatch **M, int count, int count2, int mcount, float Ts, int h,float *obj,float *outLamda,float **outY,int* outFl)
{
	try{
		typedef IloArray<IloNumVarArray> NumVarMatrix;
		IloInt i, j;
		IloModel mod(env);
		NumVarMatrix y(env, count - 1);
		for (i = 0; i < count - 1; i++)
		{
			y[i] = IloNumVarArray(env, h);
			for (j = 0; j < h; j++)
				y[i][j] = IloNumVar(env, 0.0, 1.0, ILOINT);
		}
		for (j = 0; j <= h - 1; j++)
		{
			IloExpr expr(env);
			for (i = 0; i <= count - 2; i++)
			{
				if (S_T[i + 1][j + 1] == 1)
					expr += y[i][j];
			}
			mod.add(expr == q);
			expr.end();
		}
		IloNumVarArray fl(env, count2 - 1);
		for (i = 0; i <= count2 - 2; i++)
			fl[i] = IloNumVar(env, 0.0, IloInfinity, ILOINT);
		for (i = 0; i <= count - 2; i++)
		{
			IloExpr expr1(env);
			IloExpr expr2(env);
			IloExpr expr3(env);
			for (j = 1; j <= count2 - 1; j++) {
				if (E_L[i + 1][j] == 1)
					expr1 += fl[j - 1];
				if (S_L[i + 1][j] == 1)
					expr3 += fl[j - 1];
			}
			for (j = 0; j <= h - 1; j++) {
				expr2 += y[i][j];
			}
			mod.add(expr1 + expr2 == expr3);
			//cout << expr1 << "+" << expr2 << "=" << expr3<<endl;
			expr1.end();
			expr2.end();
			expr3.end();
		}
		IloExpr expr(env);
		for (i = 1; i <= count2 - 1; i++)
		{
			if (E_L[0][i] == 1)
				expr += fl[i - 1];
		}
		mod.add(expr == h*q);
		expr.end();
		IloNumVarArray lamda(env, mcount - 1);
		for (i = 0; i <= mcount - 2; i++)
		{
			lamda[i] = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT);
		}
		for (i = 0; i <= count2 - 2; i++)
		{
			IloExpr expr(env);
			for (j = 1; j <= mcount - 1; j++)
			{
				expr += M[j][i + 1].s*R[M[j][i + 1].r] * Ts*lamda[j - 1];
			}
			mod.add(fl[i] * dp <= expr);
			expr.end();
		}
		IloExpr expr4(env);
		for (i = 0; i <= mcount - 2; i++)
		{
			expr4 += lamda[i];
		}
		mod.add(IloMinimize(env, expr4));
		expr4.end();
		IloCplex cplex(mod);
		//cplex.exportModel("diet.lp");
		cplex.setOut(env.getNullStream());
		cplex.setWarning(env.getNullStream());
		cplex.solve();
		//cplex.out() << "Solution status = " << cplex.getStatus() << endl;
		//cout << "%%%%%%%%%%%%%%%";
		*obj=cplex.getObjValue();
		for (i = 0; i <= count - 2; i++)
		for (j = 0; j <= h - 1; j++)
		{
			outY[i + 1][j + 1] = cplex.getValue(y[i][j]);
		}
		for (i = 0; i <= count2 - 2; i++)
		outFl[i+1]=cplex.getValue(fl[i]);
		for (i = 0; i <= mcount - 2; i++)
		outLamda[i+1]=cplex.getValue(lamda[i]);
		mod.end();
		cplex.end();
		y.end();
		fl.end();
		lamda.end();
	}
	catch (IloException& e) {
		cerr << "Concert exception caught: " << e << endl;
	}
	catch (...) {
		cerr << "Unknown exception caught" << endl;
	}
}

void dointlinear(IloEnv env, int **S_T, int **S_L, int **E_L, bimatch **M, int count, int count2, int mcount, float Ts, int h, int *obj, int *outLamda,int **outY,int *outFl)
{
	try{
		typedef IloArray<IloNumVarArray> NumVarMatrix;
		IloInt i, j;
		IloModel mod(env);
		NumVarMatrix y(env, count - 1);
		for (i = 0; i < count - 1; i++)
		{
			y[i] = IloNumVarArray(env, h);
			for (j = 0; j < h; j++)
				y[i][j] = IloNumVar(env, 0.0, 1.0, ILOINT);
		}
		for (j = 0; j <= h - 1; j++)
		{
			IloExpr expr(env);
			for (i = 0; i <= count - 2; i++)
			{
				if (S_T[i + 1][j + 1] == 1)
					expr += y[i][j];
			}
			mod.add(expr == q);
			expr.end();
		}
		IloNumVarArray fl(env, count2 - 1);
		for (i = 0; i <= count2 - 2; i++)
			fl[i] = IloNumVar(env, 0.0, IloInfinity, ILOINT);
		for (i = 0; i <= count - 2; i++)
		{
			IloExpr expr1(env);
			IloExpr expr2(env);
			IloExpr expr3(env);
			for (j = 1; j <= count2 - 1; j++) {
				if (E_L[i + 1][j] == 1)
					expr1 += fl[j - 1];
				if (S_L[i + 1][j] == 1)
					expr3 += fl[j - 1];
			}
			for (j = 0; j <= h - 1; j++) {
				expr2 += y[i][j];
			}
			mod.add(expr1 + expr2 == expr3);
			//cout << expr1 << "+" << expr2 << "=" << expr3<<endl;
			expr1.end();
			expr2.end();
			expr3.end();
		}
		IloExpr expr(env);
		for (i = 1; i <= count2 - 1; i++)
		{
			if (E_L[0][i] == 1)
				expr += fl[i - 1];
		}
		mod.add(expr == h*q);
		expr.end();
		IloNumVarArray lamda(env, mcount - 1);
		for (i = 0; i <= mcount - 2; i++)
		{
			lamda[i] = IloNumVar(env, 0.0, IloInfinity, ILOINT);
		}
		
		for (i = 0; i <= count2 - 2; i++)
		{
			IloExpr expr(env);
			for (j = 1; j <= mcount - 1; j++)
			{
				expr += M[j][i + 1].s*R[M[j][i + 1].r] * Ts*lamda[j - 1];
			}
			mod.add(fl[i] * dp <= expr);
			expr.end();
		}
		
		IloExpr expr4(env);
		for (i = 0; i <= mcount - 2; i++)
		{
			expr4 += lamda[i];
		}
		mod.add(IloMinimize(env, expr4));
		expr4.end();
		IloCplex cplex(mod);
		
		cplex.setOut(env.getNullStream());
		cplex.setWarning(env.getNullStream());
		//cout << Ts;
		cplex.solve();
		
		cplex.out() << "the optimal value is = " << cplex.getObjValue() << endl;
		*obj = cplex.getObjValue();
		/*for (i = 0; i <= mcount - 2; i++)
			cout<<cplex.getValue(lamda[i])<<" ";*/
		for (i = 0; i <= count - 2; i++)
		for (j = 0; j <= h - 1; j++)
		{
			outY[i + 1][j + 1] = cplex.getValue(y[i][j]);
		}
		for (i = 0; i <= count2 - 2; i++)
			outFl[i + 1] = cplex.getValue(fl[i]);
		for (i = 0; i <= mcount - 2; i++)
			outLamda[i + 1] = cplex.getValue(lamda[i]);
		
		cplex.end();
		
		//cout << "%%%%%%%%%%%%%%%";
		y.end();
		fl.end();
		lamda.end();
		mod.end();
	}
	catch (IloException& e) {
		cerr << "Concert exception caught: " << e << endl;
	}
	catch (...) {
		cerr << "Unknown exception caught" << endl;
	}
}

void dolinear(IloEnv env, IloNumArray vals, int **S_T, int **S_L, int **E_L, bimatch **M, int count, int count2, int mcount, float Ts, int h, float *obj, float *outLamda,float **outY,float *outFl)
{
	try{
		typedef IloArray<IloNumVarArray> NumVarMatrix;
		IloInt i, j;
		IloModel mod(env);
		IloRangeArray rng(env);
		NumVarMatrix y(env, count - 1);
		for (i = 0; i < count - 1; i++)
		{
			y[i] = IloNumVarArray(env, h, 0.0, 1.0);
			for (j = 0; j < h; j++)
				y[i][j] = IloNumVar(env, 0.0, 1.0, ILOFLOAT);
		}
		for (j = 0; j <= h - 1; j++)
		{
			IloExpr expr(env);
			for (i = 0; i <= count - 2; i++)
			{
				if (S_T[i + 1][j + 1] == 1)
					expr += y[i][j];
			}
			mod.add(expr == q);
			expr.end();
		}
		IloNumVarArray fl(env, count2 - 1, 0.0, IloInfinity);
		for (i = 0; i <= count2 - 2; i++)
			fl[i] = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT);
		for (i = 0; i <= count - 2; i++)
		{
			IloExpr expr1(env);
			IloExpr expr2(env);
			IloExpr expr3(env);
			for (j = 1; j <= count2 - 1; j++) {
				if (E_L[i + 1][j] == 1)
					expr1 += fl[j - 1];
				if (S_L[i + 1][j] == 1)
					expr3 += fl[j - 1];
			}
			for (j = 0; j <= h - 1; j++) {
				expr2 += y[i][j];
			}
			mod.add(expr1 + expr2 == expr3);
			//cout << expr1 << "+" << expr2 << "=" << expr3<<endl;
			expr1.end();
			expr2.end();
			expr3.end();
		}
		IloExpr expr(env);
		for (i = 1; i <= count2 - 1; i++)
		{
			if (E_L[0][i] == 1)
				expr += fl[i - 1];
		}
		mod.add(expr == h*q);
		expr.end();
		IloNumVarArray lamda(env, mcount - 1, 0.0, IloInfinity);
		for (i = 0; i <= count2 - 2; i++)
		{
			IloExpr expr(env);
			for (j = 1; j <= mcount - 1; j++)
			{
				expr += M[j][i + 1].s*R[M[j][i + 1].r] * Ts*lamda[j - 1];
			}
			expr += fl[i] * (-dp);
			rng.add(0 <= expr);
			expr.end();
		}
		mod.add(rng);
		IloExpr expr4(env);
		for (i = 0; i <= mcount - 2; i++)
		{
			expr4 += lamda[i];
		}
		mod.add(IloMinimize(env, expr4));
		expr4.end();
		IloCplex cplex(mod);
		//cplex.exportModel("diet.lp");
		cplex.setOut(env.getNullStream());
		cplex.setWarning(env.getNullStream());
		
		cplex.solve();
		cplex.out() << "Solution value = " << cplex.getObjValue() << endl;
		*obj = cplex.getObjValue();
		for (i = 0; i <= count - 2; i++)
		for (j = 0; j <= h - 1; j++)
		{
			outY[i + 1][j + 1] = cplex.getValue(y[i][j]);
		}
		for (i = 0; i <= count2 - 2; i++)
			outFl[i + 1] = cplex.getValue(fl[i]);
		for (i = 0; i <= mcount - 2; i++)
			outLamda[i+1]=cplex.getValue(lamda[i]);
		cplex.getDuals(vals, rng);
		cplex.out() << "Duals         = " << vals << endl;
		cplex.end();
		mod.end();
		rng.end();
		y.end();
		fl.end();
		lamda.end();
	}
	 catch (IloException& e) {
      cerr << "Concert exception caught: " << e << endl;
   }
   catch (...) {
      cerr << "Unknown exception caught" << endl;
   }	
}

int ScrapMatch(bimatch **M, int mcount, int count2, int *Fl,int *Start,int *End)
{
	int i, j;
	int *scrap = new int[count2];
	for (i = 1; i <= count2 - 1; i++)
		scrap[i] = 0;
	for (i = 1; i <= mcount - 1; i++)
	{
		for (j = 1; j <= count2 - 1; j++)
		{
			if (M[i][j].s)
				scrap[j]=1;
		}
	}
	int count = 1;
	for (j = 1; j <= count2 - 1; j++)
	{
		int *trans = new int[count2];
		float packet = 0;
		int j2, j3,flag2=0;
		if (scrap[j])
		{
			int k = 1;
			while (k <= mcount - 1)
			{
				for (i = k; i <= mcount - 1; i++)
				{
					int flag = 0;
					if (M[i][j].s)
					{
						packet += float(M[i][j].r * M[i][0].s) / dp;
						cout << packet << " ";
						cout << Fl[j]<<" ";
						if (packet >= Fl[j]) { flag2 = 1; break; }
						for (j2 = 1; j2 <= count2 - 1; j2++)
						{
							if (M[i][j2].s && j2 != j)
							{
								for (j3 = 1; j3 <= count - 1; j3++)
								{
									if (Start[j2] == Start[trans[j3]] || End[j2] == Start[trans[j3]])
									{
										flag = 1;
										break;
									}
								}
								if (flag) break;
							}
						}
						if (flag) packet -= float(M[i][j].r * M[i][0].s) / dp;
						else
						{
							for (j2 = 1; j2 <= count2 - 1; j2++)
							{
								if (M[i][j2].s && j2 != j)
								{
									trans[count] = j2;
									count++;
								}
							}
						}
						
					}
				}
				if (packet >= Fl[j]) break;
				k++;
				packet = 0;
			}
			if (flag2) continue;
			else return 0;
		}
		delete []trans;
		count = 1;
	}
	return 1;
}

void clear(bimatch **M, int *mcount, int count2)
{
	int i,j;
	for (i = 1; i <= *mcount - 1; i++)
	{
		int flag = 0;
		for (j = 1; j <= count2 - 1; j++)
		{
			if (M[i][j].s)
			{
				flag = 1;
				break;
			}
		}
		if (!flag)
		{
			M[i][0].s = 0;
		}
	}
	i = 1;
	int k;
	while (i <= *mcount - 1)
	{
		if (M[i][0].s == 0)
		{
			for (k = i; k <= *mcount - 2; k++)
			{
				M[k][0].s = M[k + 1][0].s;
				for (j = 1; j <= count2 - 1; j++)
				{
					M[k][j].s = M[k + 1][j].s;
					M[k][j].r = M[k + 1][j].r;
				}
			}
			(*mcount)--;
		}
		else i++;
	}
}

 void main()
{
	int i, j, count=1, count2 = 1, n, m, h;
	cin >> n >> m >> h;
	//cin >> count >> h;
	//count++;
	/*point *S = new point[count+1];
	point *T = new point[h + 1];
	link *L = new link[count*(count - 1) / 2];*/
	point *S = new point[n*m + 1];
	point *T = new point[h + 1];
	link *L = new link[n*m*(n*m - 1) / 2];
	time_t start, end;
	fstream file;
	fstream file2;
	fstream file3;
	start = time(NULL);
	file.open("C:\\Users\\popcandy\\Desktop\\wireless sensor network\\data.txt", ios::trunc | ios::out);
	file2.open("C:\\Users\\popcandy\\Desktop\\wireless sensor network\\150 targets in.txt", ios::app | ios::in);
	file3.open("C:\\Users\\popcandy\\Desktop\\wireless sensor network\\64 sensor in.txt", ios::app | ios::in);
	float aper, bper;
	aper = c / float(n + 1);
	bper = c / float(m + 1);
	//设置sensor坐标
	//for (i = 1; i <= count - 1; i++)
		//file3 >> S[i].x >> S[i].y;
	for (i = 1; i <= n; i++) {
		for (j = 1; j <= m; j++) {
			if (aper*i!= c / 2 || bper*j!= c / 2)
			{
				S[count].x = aper*i;
				S[count].y = bper*j;
				
				count++;
			}
		}
	}
	//for (i = 1; i <= 65; i++)
		//file3 << S[i].x << " " << S[i].y << endl;
	file3.close();
	S[0].x = c / 2;
	S[0].y = c / 2;
	for (i = 0; i <= count - 1; i++)
	{
		cout << i << ":(" << S[i].x << "," << S[i].y << ")" << endl;
	}
	//设置target坐标
	srand(unsigned(time(NULL)));
	file << "below are the " << h << " targets' x coordinates and y coodinates:" << endl;
	/*for (i = 1; i <= h; i++)
	{
		T[i].x=rand()%int(c);
		T[i].y = rand() % int(c);
		file2 << T[i].x << " " << T[i].y << endl;
	}*/
	for (i = 1; i <= h; i++)
	{
		file2>>T[i].x;
		file2 >> T[i].y;
		file << T[i].x << " " << T[i].y << endl;
	}
	file2.close();
	file << endl;
	trans **S_S = new trans*[count + 1];
	for (i = 1; i <= count; i++)
	{
		S_S[i] = new trans[count + 1];
	}
	for (i = 1; i <= count - 1; i++)
	{
		for (j = 0; j <= count - 1; j++)
			S_S[i][j].dist = 0;
	}
	IloNum TRmax;
	TRmax = P[plength - 1] / (Th[0] * N0);
	for (i = 0; i <= count - 1; i++) {
		for (j = 1; j <= count - 1; j++) {
			if (distance(S, S, j, i) <= TRmax && distance(S, S, i, 0) < distance(S, S, j, 0))
			{
				L[count2].s = j;
				L[count2].e = i;
				L[count2].num = count2;
				S_S[j][i].dist = distance(S, S, j, i);
				S_S[j][i].num = count2;
				count2++;
			}
			else
			{
				S_S[j][i].dist = distance(S, S, j, i);
				S_S[j][i].num = 0;
			}
		}
	}
	cout << endl;
	int **S_L = new int*[count+1];
	int **E_L = new int*[count+1];
	int *End_Search = new int [count2+1];
	int *Start_Search = new int[count2 + 1];
	for (i = 0; i <= count - 1; i++)
	{
		S_L[i] = new int[count2+1];
		E_L[i] = new int[count2 + 1];
	}
	for (i = 0; i <= count - 1;i++)
	for (j = 1; j <= count2 - 1; j++)
	{
		S_L[i][j] = 0;
		E_L[i][j] = 0;
	}
	for (i = 1; i <= count2 - 1; i++)
	{
		S_L[L[i].s][i] = 1;
		Start_Search[i] = L[i].s;
		E_L[L[i].e][i] = 1;
		End_Search[i] = L[i].e;
	}
	/*for (i = 0; i <= count - 1; i++)
	{
		cout << i << "->";
		for (j = 1; j <= count2 - 1; j++)
		{
			if (S_L[i][j])
				cout << j << " ";
		}
		cout << endl;
	}
	for (i = 0; i <= count - 1; i++)
	{
		for (j = 1; j <= count2 - 1; j++)
		{
			if (E_L[i][j])
				cout << j << " ";
		}
		cout << "->" << i;
		cout << endl;
	}*/
	//sensor与可检测target的匹配对
	int **S_T = new int*[count + 1];
	for (i = 0; i <= count-1; i++)
	{
		S_T[i] = new int[h + 1];
	}
	for (i = 0; i <= count - 1; i++)
	for (j = 1; j <= h; j++)
	{
		S_T[i][j] = 0;
	}
	for (i = 1; i <= count - 1; i++)
	for (j = 1; j <= h; j++)
	if (distance(S, T, i, j) <= Rmax*Rmax)
	{
		S_T[i][j] = 1;
	}
	/*for (i = 1; i <= count - 1; i++)
	{
		for (j = 1; j <= h; j++)
			cout << S_T[i][j] << " ";
		cout << endl;
	}*/
	link *L2 = new link[count2+1];
	int *q1 = new int[count + 1];
	int count3 = 1;
	int *taken = new int[count + 1];
	for (i = 1; i <= count - 1; i++) taken[i] = 0;
	int head = 0, tail = 1;
	q1[0] = 0;
	while (head < tail)
	{
		for (i = 1; i <= count - 1; i++)
		if (!taken[i] && S_S[i][q1[head]].num)
		{
			L2[count3].s = i;
			L2[count3].e = q1[head];
			L2[count3].num = S_S[i][q1[head]].num;
			count3++;
			q1[tail] = i;
			tail++;
			taken[i] = 1;
		}
		head++;

	}
	/*cout << count3<<endl;
	for (i = 1; i <= count3 - 1; i++)
	{
		cout <<L2[i].num<<":"<< L2[i].s << "->" << L2[i].e << endl;
	}*/
	delete[] q1;
	delete[] taken;
	int *taken2 = new int[count3 + 1];
	for (i = 1; i <= count3 - 1; i++)
	{
		taken2[i] = 0;
	}

	float Ts = dp / (R[0]*2.0);
	float Ts2 = dp / (R[0]);
	vertice *L3 = new vertice[count3+1];
	for (i = 1; i <= count3 - 1; i++)
	{
		L3[i].s = L2[i].s;
		L3[i].e = L2[i].e;
		L3[i].r = rlength - 4;
		L3[i].num = L2[i].num;
	}
	for (i = 1; i <= count3 - 1; i++)
	{
		cout << L3[i].num << ":" << L3[i].s << "->" << L3[i].e << "   ";
	}
	/*for (i = 1; i <= count3 - 1; i++)
	{
		cout << L3[i].s << "->" << L3[i].e <<" "<< L3[i].r <<" "<<L3[i].num<<endl;
	}*/
	double *w = new double[count3];
	for (i = 1; i <= count3 - 1; i++)
		w[i] = 0.0041;
	bimatch **M;
	M = (bimatch **)malloc(2 * sizeof(bimatch *));
	M[1] = (bimatch *)malloc((count2 + 1)*sizeof(bimatch));
	for (j = 1; j <= count2 - 1; j++)
	{
		M[1][j].s = 0;
		M[1][j].r = 0;
	}
	int mcount = 1;
	cout << count3<<endl;
	for (i = 1; i <= count3 - 1; i++)
	{
		if (!taken2[i])
		{
			if (mgeneration(L3, count3, w, taken2, M, mcount, S_S,count2))
			{
				mcount++;
				M = (bimatch **)realloc(M, (mcount + 1)*sizeof(bimatch *));
				M[mcount] = (bimatch *)malloc((count2 + 1)*sizeof(bimatch));
				for (j = 1; j <= count2 - 1; j++)
				{
					M[mcount][j].s = 0;
					M[mcount][j].r = 0;
				}
			}
		}
	}
	/*for (i = 1; i <= count3 - 1; i++){
		cout << taken2[i] << " ";
	}
	cout << endl;*/
	delete []L3;
	delete []w;
	delete []taken2;
	/*for (i = 1; i <= mcount - 1; i++)
	{
		for (j = 1; j <= count2 - 1; j++)
		{
			cout << M[i][j].s;
		}
		cout << endl;
	}
	cout << endl;
	for (i = 1; i <= mcount - 1; i++)
	{
		for (j = 1; j <= count2 - 1; j++)
		{
			cout << M[i][j].r;
		}
		cout << endl;
	}*/
	em = 1; 
	//定义用于输出到文件的变量
	int IntObj;
	int **IntY = new int *[count];
	for (i = 1; i <= count-1; i++)
	{
		IntY[i] = new int[h+1];
	}
	int *IntFl = new int[count2];
	int *IntLamda=new int[mcount];

	float Obj;
	float **Y = new float *[count];
	for (i = 1; i <=count-1; i++)
	{
		Y[i] = new float[h+1];
	}
	float *Fl = new float[count2];
	float *Lamda=new float[mcount];
	float MixObj;
	float **MixY = new float *[count];
	for (i = 0; i <= count-1; i++)
	{
		MixY[i] = new float[h+1];
	}
	int *MixFl = new int[count2];
	float *MixLamda = new float[mcount];
	while (em)
	{
		IloEnv env2;
		IloNumArray vals(env2);
		dolinear(env2, vals, S_T, S_L, E_L, M, count, count2, mcount, Ts2, h, &Obj, Lamda, Y, Fl);
		IloEnv env;
		dointlinear(env, S_T, S_L, E_L, M, count, count2, mcount, Ts, h, &IntObj, IntLamda, IntY, IntFl);
		env.end();
		vertice *L3 = new vertice[rlength*(count2 + 1)];
		count3 = 1;
		double *w = new double[rlength*(count2 + 1)];
		for (i = 1; i <= count2 - 1; i++)
		for (j = 0; j <= rlength - 1; j++)
		{
			if (vals[i - 1] != 0)
			{
				L3[count3].s = L[i].s;
				L3[count3].e = L[i].e;
				L3[count3].r = j;
				L3[count3].num = L[i].num;
				w[count3] = vals[i - 1] * Ts2;
				count3++;
			}
		}
		int *taken = new int[count3 + 1];
		for (i = 1; i <= count3 - 1; i++)
			taken[i] = 0;
		if (mgeneration(L3, count3, w, taken, M, mcount, S_S,count2))
		{
			mcount++;
			M = (bimatch **)realloc(M, (mcount + 1)*sizeof(bimatch *));
			M[mcount] = (bimatch *)malloc((count2 + 1)*sizeof(bimatch));
			IntLamda = (int *)realloc(IntLamda, (mcount + 1)*sizeof(int));
			MixLamda = (float *)realloc(MixLamda, (mcount + 1)*sizeof(float));
			Lamda = (float *)realloc(Lamda, (mcount + 1)*sizeof(float));
			for (j = 1; j <= count2 - 1; j++)
			{
				M[mcount][j].s = 0;
				M[mcount][j].r = 0;
			}
			em = 1;
		}
		else em = 0;
		for (i = 1; i <= mcount - 1; i++)
		{
			cout << "the " << i << "th schedulings are:";
			for (j = 1; j <= count2 - 1; j++)
			{
				if (M[i][j].s)
				{
					cout<< j << ":r=" << R[M[i][j].r] << " ";
				}
			}
			cout << endl;
		}
		vals.end();
		env2.end();
		delete[]L3;
		delete[]w;
		delete[]taken;
	}
	end = time(NULL);
	cout << difftime(end,start)<<endl;
	int totalbit=0;
	int totalp=0;
	for (i = 1; i <= mcount-1; i++)
	for (j = 1; j <= count2 - 1; j++)
	{
		if (M[i][j].s)
			totalbit += IntLamda[i] * Ts * R[M[i][j].r];
	}
	for (j = 1; j <= count2 - 1; j++)
		totalp += IntFl[j] * dp;
	//将数据输出到文件
	file <<"the total number of schedulings"<< mcount - 1 << endl;
	//int linear programming 1
	file <<"The int optimal value 1 is:"<< IntObj << endl;
	file << "Throughput is:" << h*q*dp/(IntObj*Ts)<< endl;
	file << "The delay is:" << IntObj*Ts << endl;
	cout << totalbit << endl;
	file << "The waste bits:"<<totalbit-totalp<<endl;
	file << "The sensing situation is:" << endl;
	for (i = 1; i <= count - 1; i++)
	{
		file << "sensor" << i << " is in charge of target:";
		for (j = 1; j <= h; j++)
		{
			if (IntY[i][j])
			file<<j<<" ";
		}
		file << endl;
	}
	file << "The link carrying situation is:" << endl;
	for (i = 1; i <= count2 - 1; i++)
	{
		file << "link" << i << " carrys "<<IntFl[i]<<" data packet";
		file << endl;
	}
	file << "schedulings are:" << endl;
	for (i = 1; i <= mcount - 1; i++)
	{
		if (IntLamda[i])
		file <<i<<" th schedulings schedules "<< IntLamda[i] << "times "<<endl;
	}
	file << endl;
	//mix linear programming
	/*file << "The mix optimal value is:" << MixObj << endl;
	file << "The sensing situation is:" << endl;
	for (i = 1; i <= count - 1; i++)
	{
		file << "sensor" << i << " is in charge of target:";
		for (j = 1; j <= h; j++)
		{
			if (MixY[i][j])
				file << j << " ";
		}
		file << endl;
	}
	file << "The link carrying situation is:" << endl;
	for (i = 1; i <= count2 - 1; i++)
	{
		file << "link" << i << " carrys " << MixFl[i] << " data packet";
		file << endl;
	}
	file << "schedulings are:" << endl;
	for (i = 1; i <= mcount - 1; i++)
	{
		if (MixLamda[i])
			file << i << " th schedulings schedules " << MixLamda[i] << " times " << endl;
	}
	file << endl;*/
	//normal linear programming
	file << "The normal optimal value is:" << Obj << endl;
	file << "The sensing situation is:" << endl;
	for (i = 1; i <= count - 1; i++)
	{
		file << "sensor" << i << " is in charge of target:";
		for (j = 1; j <= h; j++)
		{
			if (Y[i][j])
				file << j << " ";
		}
		file << endl;
	}
	file << "The link carrying situation is:" << endl;
	for (i = 1; i <= count2 - 1; i++)
	{
		file << "link" << i << " carrys " << Fl[i] << " data packet";
		file << endl;
	}
	file << "schedulings are:" << endl;
	for (i = 1; i <= mcount - 1; i++)
	{
		if (Lamda[i])
		file << i << " th schedulings schedules " << Lamda[i] << " times " << endl;
	}
	file << endl;
	for (i = 1; i <= mcount - 1; i++)
	{
		file << "the "<<i<<"th schedulings are:";
		for (j = 1; j <= count2 - 1; j++)
		{
			if (M[i][j].s)
			{
				file << j << ":r=" << R[M[i][j].r] << " ";
			}
		}
		file << endl;
	}
	file.close();

	//计算碎片
	bimatch **MissMatch;
	MissMatch = (bimatch **)malloc(2 * sizeof(bimatch *));
	MissMatch[1] = (bimatch *)malloc((count2 + 1)*sizeof(bimatch));
	for (i = 1; i <= count2 - 1; i++)
		MissMatch[1][i].s = 0;
	int MisCount = 1;
	for (i = 1; i <= mcount; i++)
	{
		int sched = 10000;
		int flag = 0;
		for (j = 1; j <= count2 - 1; j++)
		{
			int nsched = 0;
			if (M[i][j].s && int(IntLamda[i] * (Ts) * R[M[i][j].r]) % int(dp) && IntFl[j])
			{
				int tmp = int(IntLamda[i] * (Ts) * R[M[i][j].r]) / int(dp);
				if (tmp < IntFl[j])
				{
					flag = 1;
					MissMatch[MisCount][j].s = 1;
					nsched = tmp * int(dp) / (R[M[i][j].r]*(Ts));
					if (nsched < sched) sched = nsched;
				}
			}
		}
		if (flag)
		{
			IntLamda[i] -= sched;
			for (j = 1; j <= count2 - 1; j++)
			{
				if (int(sched * (Ts) * R[M[i][j].r]) / int(dp) > IntFl[j])
					IntFl[j] = 0;
				else
				{
					IntFl[j] -= int(sched * (Ts)* R[M[i][j].r]) / int(dp);
				}
					
			}
			MissMatch[MisCount][0].s = IntLamda[i];
			for (j = 1; j <= count2 - 1; j++)
			{
				if (M[i][j].s)
				{
					MissMatch[MisCount][j].r = R[M[i][j].r];
				}
				else
					MissMatch[MisCount][j].r = 0;
			}
			MisCount++;
			MissMatch = (bimatch **)realloc(MissMatch, (MisCount + 1)*sizeof(bimatch *));
			MissMatch[MisCount] = (bimatch *)malloc((count2 + 1)*sizeof(bimatch));
			for (j = 1; j <= count2 - 1; j++)
				MissMatch[MisCount][j].s = 0;
		}
		else
		{
			for (j = 1; j <= count2 - 1; j++)
			{
				if (int(IntLamda[i] * (Ts) * R[M[i][j].r]) / int(dp) > IntFl[j])
					IntFl[j] = 0;
				else
					IntFl[j] -= int(IntLamda[i] * (Ts) * R[M[i][j].r]) / int(dp);
			}
			IntLamda[i] = 0;
		}

	}
	//请出已调度完成的边
	for (j = 1; j <= count2-1; j++)
	{
		int count = 0;
		int flag;
		for (i = 1; i <= MisCount - 1; i++)
		if (MissMatch[i][j].s)
		{
			count++;
			flag = i;
		}
		if (count == 1)
			MissMatch[flag][j].s = 0;
	}

	cout << MisCount << endl;
	for (i = 1; i <= MisCount - 1; i++)
	{
		cout << MissMatch[i][0].s << endl;
		if (MissMatch[i][0].s)
		{
			for (j = 1; j <= count2 - 1; j++)
			if (MissMatch[i][j].r)
				cout << j << ":" << MissMatch[i][j].s << " " << MissMatch[i][j].r << "  ";
			cout << endl;
		}
	}

	/*for (i = 1; i <= MisCount - 1; i++)
	{
		int flag = 0;
		for (j = 1; j <= count2 - 1; j++)
		{
			if (MissMatch[i][j].s)
			{
				flag = 1;
				break;
			}
		}
		if (!flag)
		{
			MissMatch[i][0].s = 0;
		}
	}
	i = 1;
	int k;
	while (i <= MisCount - 1)
	{
		if (MissMatch[i][0].s == 0)
		{
			for (k = i; k <= MisCount - 2; k++)
			{
				MissMatch[k][0].s = MissMatch[k + 1][0].s;
				for (j = 1; j <= count2 - 1; j++)
				{
					MissMatch[k][j].s = MissMatch[k + 1][j].s;
					MissMatch[k][j].r = MissMatch[k + 1][j].r;
				}
			}
			(MisCount)--;
		}
		else i++;
	}*/
	clear(MissMatch, &MisCount, count2);
	cout << MisCount << endl;
	for (i = 1; i <= MisCount - 1; i++)
	{
		cout << MissMatch[i][0].s << endl;
		if (MissMatch[i][0].s)
		{
			for (j = 1; j <= count2 - 1; j++)
			if (MissMatch[i][j].r)
				cout << j << ":" << MissMatch[i][j].s<<" "<< MissMatch[i][j].r << "  ";
			cout << endl;
		}
	}

	//碎片整理
	if ((MisCount-1))
	{
		if (ScrapMatch(MissMatch,MisCount,count2,IntFl,Start_Search,End_Search))
			cout << "Successfully scheduled!";//碎片整理成功
		else cout << "Failed scheduled!";//整理失败
	}
	else
	{
		cout << "Successfully scheduled!";//无碎片
	}

}