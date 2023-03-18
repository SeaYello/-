#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
#define rep(x,y,z) for(ll x=y;x<=z;x++)
#define xa (x+a[i])
#define yb (y+b[i])
struct point{ll x, y;};

void xin(int &n) {scanf("%d", &n);}
void xin(ll &n) {scanf("%ld", &n);}
void xin(float &n) {scanf("%f", &n);}
void xin(double &n) {scanf("%lf", &n);}
void xin(string &s) {cin>>s;}
void xin(char &c) {c=getchar();}
void xout(int n) {printf("%d", n);}
void xout(ll n) {printf("%ld", n);}
void xout(float n) {printf("%f", n);}
void xout(float n, int d) {char s[]=".0f";s[1]=d+'0';printf(s, n);}
void xout(double n) {printf("%lf", n);}
void xout(double n, int d) {char s[]=".0lf";s[1]=d+'0';printf(s, n);}
void xout(const string &s) {printf("%s", &(s[0]));}
void xout(const char *s) {printf("%s", s);}
void xout(char c) {putchar(c);}
void xendl() {putchar('\n');}

template<typename T> T bisel(bool (*f)(T x), T l, T r) {
    while(r-l>1) {
        ll m=(l+r)/2;
        (f(l)==f(m)?l:r) = m;
    } return l;
}

template<typename T> T fac(T n) {
    return n?n*fac(n-1):1;
}

template<typename T> void presum(vector<T> &v) {
    for(int i=1;i<v.size();i++) v[i]+=v[i-1];
}

template<typename T, typename P> int find(T &s, P e) {
    for(int i=0;i<s.size();i++) if(s[i]==e) return i; return -1;
}

template<typename T> void fill4(vector<vector<T>> &v, int x, int y, T e) {
    int n=v.size(), m=v[0].size();
    T ori = v[x][y];
    v[x][y] = e;
    if(x>0&&v[x-1][y]==ori) fill4(v,x-1,y,e);
    if(y>0&&v[x][y-1]==ori) fill4(v,x,y-1,e);
    if(x<n-1&&v[x+1][y]==ori) fill4(v,x+1,y,e);
    if(y<m-1&&v[x][y+1]==ori) fill4(v,x,y+1,e);
}

template<typename T> void fill8(vector<vector<T>> &v, int x, int y, T e) {
    int n=v.size(), m=v[0].size();
    T ori = v[x][y];
    v[x][y] = e;
    if(x>0&&v[x-1][y]==ori) fill8(v,x-1,y,e);
    if(y>0&&v[x][y-1]==ori) fill8(v,x,y-1,e);
    if(x<n-1&&v[x+1][y]==ori) fill8(v,x+1,y,e);
    if(y<m-1&&v[x][y+1]==ori) fill8(v,x,y+1,e);
    if(x>0&&y>0&&v[x-1][y-1]==ori) fill8(v,x-1,y-1,e);
    if(x<n-1&&y>0&&v[x+1][y-1]==ori) fill8(v,x+1,y-1,e);
    if(x>0&&y<m-1&&v[x-1][y+1]==ori) fill8(v,x-1,y+1,e);
    if(x<n-1&&y<m-1&&v[x+1][y+1]==ori) fill8(v,x+1,y+1,e);
}

ll qpow(ll base,ll k,ll mod) {
	ll tmp=1; 
	for(;k;base=base*base%mod,k>>=1) if(k&1) tmp=tmp*base%mod; 
	return tmp; 
}

bool is_prime(ll x)  {
    int arr[10]={2,3,5,7,11,13,17,23};
	if(x<=1) return 0; 
	int i,j,k; 
	ll pre,a,cur;  
	for(i=0;i<8;++i) 
	{
		if(x==arr[i]) return 1;          
		for(cur=x-1,k=0;cur%2==0;cur>>=1) ++k;              
		pre=a=qpow(arr[i],cur,x);              
		for(j=1;j<=k;++j) 
		{  
			a=(a*a)%x;                 
			if(a==1&&pre!=1&&pre!=x-1) return 0; 
			pre=a;   
		}
		if(a!=1) return 0;    
	}   
	return 1;   
}

/////////////////////////////////////////////////////