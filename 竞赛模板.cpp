#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
#define rep(x,y,z) for(ll x=y;x<=z;x++)
#define xa (x+a[i])
#define yb (y+b[i])
struct point{
    ll x, y;
    bool operator<(const point &a) const { return x<a.x?1:(y<a.y); }
};

void xin(int &n) {scanf("%d", &n);}
void xin(ll &n) {scanf("%lld", &n);}
void xin(float &n) {scanf("%f", &n);}
void xin(double &n) {scanf("%lf", &n);}
void xin(string &s) {cin>>s;}
void xin(char &c) {c=getchar();}
void xout(int n) {printf("%d", n);}
void xout(ll n) {printf("%lld", n);}
void xout(ull n) {printf("%llu", n);}
void xout(float n) {printf("%f", n);}
void xout(float n, int d) {char s[]="%.0f";s[2]=d+'0';printf(s, n);}
void xout(double n) {printf("%lf", n);}
void xout(double n, int d) {char s[]="%.0lf";s[2]=d+'0';printf(s, n);}
void xout(const string &s) {printf("%s", &(s[0]));}
void xout(const char *s) {printf("%s", s);}
void xout(char c) {putchar(c);}
#define xendl putchar('\n')
template<typename T> void xout(const vector<T> &v) {for(auto e:v)xout(e),xout(',');xendl;}

vector<vector<ll>> read_mat(int n, int m) {
    vector<vector<ll>> ret;
    rep(i,0,n-1) rep(j,0,m-1) xin(ret[i][j]);
    return ret;
}

template<typename T> T bisel(bool (*f)(T x), T l, T r) {
    while(r-l>1) {
        ll m=(l+r)/2;
        (f(l)==f(m)?l:r) = m;
    } return l;
}

template<typename T> T fac(T n) { return n?n*fac(n-1):1;}

template<typename T> void presum(vector<T> &v) {rep(i,1,v.size()-1) v[i]+=v[i-1];}

template<typename T, typename P> int find(T &s, P e) {rep(i,0,s.size()-1) if(s[i]==e) return i; return -1;}

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

bool is_prime(ll x) {
    int arr[] {2,3,5,7,11,13,17,23};
	if(x<=1) return 0; 
	int k; 
	ll pre,a,cur;  
	rep(i,0,7) {
		if(x==arr[i]) return 1;          
		for(cur=x-1,k=0;cur%2==0;cur>>=1) ++k;              
		pre=a=qpow(arr[i],cur,x);              
		rep(j,1,k) {  
			a=(a*a)%x;                 
			if(a==1&&pre!=1&&pre!=x-1) return 0; 
			pre=a;   
		}
		if(a!=1) return 0;    
	}   
	return 1;   
}

vector<ll> get_primes(int n) {
    vector<ll> ret;
    vector<int> check(n+1);
    rep(i,2,n) {
        if(!check[i]) ret.push_back(i);
        rep(j,0,ret.size()-1) {
            if(i*ret[j]>n) break;
            check[i*ret[j]]=1;
            if(i%ret[j]==0) break;
        }
    }
    return ret;
}

template<typename T> vector<T> dedup(vector<T> v) {
    set<T> s(v.begin(),v.end());
    return vector<T>(s.begin(),s.end());
}

string dedup(string v) {
    set<char> s(v.begin(),v.end());
    return string(s.begin(),s.end());
}

int log2(ll x) {
    int ret=0;
    while(x>>=1) ret++;
    return ret;
}

template<typename T> vector<vector<T>> get_st(vector<T> &v, T (*sel)(T a, T b)) {
    int n = v.size(), m = log2(n)+1;
    vector<vector<T>> ret(m);
    ret[0] = v;
    rep(i,1,m-1) rep(j,0,n-(1<<i)) ret[i].push_back(sel(ret[i-1][j],ret[i-1][j+(1<<(i-1))]));
    return ret;
}

template<typename T> T st_queue(vector<vector<T>> &st, int l, int r, T (*sel)(T a, T b)) {
    int lay = log2(r-l+1), wid=1<<lay;
    return sel(st[lay][l],st[lay][r-wid+1]);
}

template<typename T> T backpack(T tv, vector<T> &w, vector<T> &v) {
    vector<ll> dp(tv+1);
    rep(i,0,w.size()-1) {
        for(int j=tv;j>=0;j--) {
            ll a = dp[j], b = j-w[i]<0?0:dp[j-w[i]]+v[i];
            dp[j]=max(a,b);
        }
    }
    return dp.back();
}

vector<ull> _str2ull(const string &s) {
    int n=s.size();
    int beg=n%9==0?9:n%9;
    vector<ull> ret(ceilf(n/9.0f));
    rep(i,0,beg-1) ret[0]=ret[0]*10+s[i]-'0';
    rep(i,beg,n-1) {
        int x=1+(i-beg)/9;
        ret[x]=ret[x]*10+s[i]-'0';
    }
    return ret;
}

string _ull2str(const vector<ull> &v) {
    string ret;
    ret.reserve(v.size()*9);
    ull f=v[0];
    do {
        ret+=f%10+'0';
    } while(f/=10);
    reverse(ret.begin(),ret.end());
    rep(i,1,v.size()-1) {
        f=v[i];
        ret+=f/100000000%10+'0';
        ret+=f/10000000%10+'0';
        ret+=f/1000000%10+'0';
        ret+=f/100000%10+'0';
        ret+=f/10000%10+'0';
        ret+=f/1000%10+'0';
        ret+=f/100%10+'0';
        ret+=f/10%10+'0';
        ret+=f/1%10+'0';
    }
    return ret;
}

vector<ull> _xadd(const vector<ull> &a, const vector<ull> &b) {
    vector<ull> ret;
    ret.reserve(max(a.size(),b.size())+1);
    ull carry=0;
    int i=a.size()-1, j=b.size()-1;
    while(i>=0||j>=0) {
        ull d1=i<0?0:a[i], d2=j<0?0:b[j], sum=d1+d2+carry;
        ret.push_back(sum%1000000000);
        carry = sum/1000000000;
        i--, j--;
    }
    if(carry) ret.push_back(carry);
    reverse(ret.begin(),ret.end());
    return ret;
}

string xadd(const string &a, const string &b) {
    string ret;
    ret.reserve(max(a.size(),b.size())+1);
    int carry=0, i=a.size()-1, j=b.size()-1;
    while(i>=0||j>=0) {
        int d1=i<0?0:a[i]-'0', d2=j<0?0:b[j]-'0', sum=d1+d2+carry;
        ret.push_back(sum%10+'0');
        carry = sum/10;
        i--,j--;
    }
    if(carry) ret += '1';
    reverse(ret.begin(),ret.end());
    return ret;
}

vector<ull> _xmul(vector<ull> v, ull n) {
    ull carry=0,sum;
    for(int i=v.size()-1;i>=0;i--) {
        sum=v[i]*n+carry;
        v[i]=sum%1000000000;
        carry=sum/1000000000;
    }
    if(carry) v.insert(v.begin(), carry);
    return v;
}

string xmul(const string &a, ull n) {
    return _ull2str(_xmul(_str2ull(a),n));
}

vector<ull> _xmul(const vector<ull> &a, const vector<ull> &b) {
    vector<ull> ret{0}, tmp;
    rep(i,0,b.size()-1) {
        tmp=_xmul(a,b[i]);
        ret=_xadd(ret,tmp);
        ret.push_back(0);
    }
    ret.pop_back();
    return ret;
}

string xmul(const string &a, const string &b) {
    return _ull2str(_xmul(_str2ull(a),_str2ull(b)));
}

/////////////////////////////////////////////////////
