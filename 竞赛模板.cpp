#include<iostream>
#include<vector>
#include<stack>
#include<set>
#include<unordered_set>
#include<map>
#include<unordered_map>
#include<numeric>
#include<math.h>
#include<algorithm>
#include<queue>
#include<string>
#include<random>
#include<time.h>
#include<cstring>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;
#define endl '\n'

//////// DATA STRUCTURE ////////

// sparse table

template<typename T> vector<vector<T>> get_st(vector<T> &v, T (*sel)(T a, T b)) {
    int n = v.size(), m = log2d(n)+1;
    vector<vector<T>> ret(m);
    ret[0] = v;
    for(int i=1;i<m;i++) for(int j=0;j<=n-(1<<i);j++) ret[i].push_back(sel(ret[i-1][j],ret[i-1][j+(1<<(i-1))]));
    return ret;
}

template<typename T> T st_queue(vector<vector<T>> &st, int l, int r, T (*sel)(T a, T b)) {
    int lay = log2d(r-l+1), wid=1<<lay;
    return sel(st[lay][l],st[lay][r-wid+1]);
}

// segment tree (min query & add modify)

struct Seg {
    ll val=1e12,lazy=0;
    int l=0,r=0;
};

int n;
vector<ll> v;
vector<Seg> seg;

void build_seg(int idx,int l,int r) {

    seg[idx].l=l;
    seg[idx].r=r;
    if(l==r) {
        seg[idx].val=v[l];
    } else {
        int mid=(l+r)/2;
        build_seg(idx*2+1,l,mid);
        build_seg(idx*2+2,mid+1,r);
        seg[idx].val=min(seg[idx*2+1].val,seg[idx*2+2].val);
    }
}

void update_seg(int idx) {
    while(idx) {
        idx=(idx-1)/2;
        seg[idx].val=min(seg[idx*2+1].val,seg[idx*2+2].val);
    }
}

void push_seg(int idx) {
    if(seg[idx].l<seg[idx].r) {
        seg[idx*2+1].lazy+=seg[idx].lazy;
        seg[idx*2+1].val+=seg[idx].lazy;
        seg[idx*2+2].lazy+=seg[idx].lazy;
        seg[idx*2+2].val+=seg[idx].lazy;
    }
    seg[idx].lazy=0;
}

void add_seg(int idx,int l,int r,ll x) {
    if(l>r) return;
    if(seg[idx].l==l && seg[idx].r==r) {
        seg[idx].val+=x;
        seg[idx].lazy+=x;
        update_seg(idx);
    } else {
        int mid=(seg[idx].l+seg[idx].r)/2;
        add_seg(idx*2+1,l,min(r,mid),x);
        add_seg(idx*2+2,max(l,mid+1),r,x);
    }
}

ll min_seg(int idx,int l,int r) {
    if(l>r) return 1e12;
    if(seg[idx].l==l && seg[idx].r==r) {
        return seg[idx].val;
    }
    push_seg(idx);
    int mid=(seg[idx].l+seg[idx].r)/2;
    return min(min_seg(idx*2+1,l,min(r,mid)),min_seg(idx*2+2,max(l,mid+1),r));
}

// dijkstra

int n;

double dijkstra(vector<vector<pair<int,double>>> g,int s,int t) {
    vector<double> dp(n,1e12); dp[s]=0;
    vector<int> vis(n);
    priority_queue<pair<double,int>,vector<pair<double,int>>,greater<pair<double,int>>> pq; pq.push({0,s});
    while(pq.size()) {
        int curr=pq.top().second; pq.pop();
        if(vis[curr]++) continue;
        for(auto [nv,w]:g[curr]) {
            dp[nv]=min(dp[nv],dp[curr]+w);
            pq.push({dp[nv],nv});
        }
    }
    return dp[t];
}

// bellman_ford

int n;

double bellman_ford(vector<vector<pair<int,double>>> g,int s,int t) {
    vector<double> dp(n,1e12);
    dp[s]=0;
    for(int i=0;i<n-1;i++) {
        for(int j=0;j<n;j++) {
            for(auto [nv,w]:g[j]) {
                dp[nv]=min(dp[nv],dp[j]+w);
            }
        }
    }
    return dp[t];
}

//////// STRING /////////

vector<int> kmp_next(const string &t) {
	int n=t.size()+1;
	vector<int> ret(n);
	for(int i=2;i<n;i++) {
		int j=ret[i-1];
		while(t[i-1]!=t[j] && j) {
			j=ret[j];
		}
		ret[i]=j+(t[i-1]==t[j]);
	}
	return ret;
}

int kmp_first(const string &s, const string &t) {
	int n=s.size(),m=t.size(),i=0,j=0;
	auto ne=kmp_next(t);
	while(i<n) {
		if(s[i]==t[j]) {
			i++;j++;
			if(j==m) {
				return i-m;
			}
		} else {
			j=ne[j];
		}
	}
	return -1;
}

vector<int> kmp_all(const string &s, const string &t) {
	int n=s.size(),m=t.size(),i=0,j=0;
	auto ne=kmp_next(t);
	vector<int> ret;
	while(i<n) {
		if(s[i]==t[j]) {
			i++;j++;
			if(j==m) {
				ret.push_back(i-m);
				j=0;
			}
		} else {
			j=ne[j];
		}
	}
	return ret;
}

vector<int> kmp_all_overlap(const string &s, const string &t) {
	int n=s.size(),m=t.size(),i=0,j=0;
	auto ne=kmp_next(t);
	vector<int> ret;
	while(i<n) {
		if(s[i]==t[j]) {
			i++;j++;
			if(j==m) {
				ret.push_back(i-m);
				j=0;
				i=i-m+1;
			}
		} else {
			if(j) j=ne[j];
			else i++;
		}
	}
	return ret;
}

///////// MATH //////////

ll MOD;

ll qpow(ll a, ll n) {
	ll ret=1;
	while(n) {
		if(n%2) ret=ret*a%MOD;
		a=a*a%MOD;
		n/=2;
	}
	return ret;
}

ll C(ll n, ll m) {
	ll a=1,b=1;
	for(int i=1;i<=m;i++) {
		a=a*(n-i+1)%MOD;
		b=b*i%MOD;
	}
	return a*qpow(b,MOD-2)%MOD;
}

bool miller_rabin(ll n) {
	if(n<=1) return false;
	int p[] {2,3,5,7};
	for(auto a:p) {
		if(a>=n) break;
		int k=0;
		ll d=n-1;
		while(d%2==0) {
			k++;
			d/=2;
		}
		ll x=qpow(a,d);
		while(k--) {
			ll y=x*x%n;
			if(y==1 && x!=1 && x!=n-1) return false;
			x=y;
		}
		if(x!=1) return false;
	}
	return true;
}

vector<ll> linear_sieve(int n) {
    vector<ll> ret;
    vector<int> check(n+1);
    for(int i=2;i<=n;i++) {
        if(!check[i]) ret.push_back(i);
        for(int j=0;j<ret.size();j++) {
            if(i*ret[j]>n) break;
            check[i*ret[j]]=1;
            if(i%ret[j]==0) break;
        }
    }
    return ret;
}

// FFT

struct CN {
    ld a=0,b=0;
};

CN operator+(CN a, CN b) {
    a.a+=b.a, a.b+=b.b;
    return a;
}

CN operator*(CN a, CN b) {
    CN c {a.a*b.a-a.b*b.b, a.a*b.b+a.b*b.a};
    return c;
}

const ld PI=3.14159265;
vector<CN> FFT(vector<CN> v) {
    int n=v.size();
    if(n==1) return v;
    vector<CN> v1,v2;
    for(int i=0;i<n;i++) {
        (i%2?v1:v2).push_back(v[i]);
    }
    v1=FFT(v1),v2=FFT(v2);
    for(int i=0;i<n;i++) {
        CN co {cosl(2*PI*i/n),sinl(2*PI*i/n)};
        v[i]=v1[i%(n/2)]*co+v2[i%(n/2)];
    }
    return v;
}

vector<CN> invFFT(vector<CN> v) {
    reverse(v.begin(),v.end());
    v=FFT(v);
    int n=v.size();
    for(int i=0;i<n;i++) {
        CN m{cosl(2*PI*i/n)/n,sinl(2*PI*i/n)/n};
        v[i]=v[i]*m;
    }
    return v;
}

vector<ll> FFT_poly(vector<ll> a,vector<ll> b) {
    int n=1<<(int)ceil(log2(a.size()+b.size()));
    a.resize(n),b.resize(n);
    vector<CN> A(n),B(n);
    for(int i=0;i<n;i++) {
        A[i].a=a[i];
        B[i].a=b[i];
    }
    A=FFT(A);
    B=FFT(B);
    for(int i=0;i<n;i++) {
        A[i]=A[i]*B[i];
    }
    A=invFFT(A);
    for(int i=0;i<n;i++) {
        a[i]=roundl(A[i].a);
    }
    while(!a.back() && a.size()>1) a.pop_back();
    return a;
}

string FFT_num(string a,string b) {
    int n=a.size(),m=b.size();
    vector<ll> A(n),B(m);
    for(int i=0;i<n;i++) A[i]=a[n-1-i]-'0';
    for(int i=0;i<m;i++) B[i]=b[m-1-i]-'0';
    A=FFT_poly(A,B);
    A.resize(n=A.size()+10);
    for(int i=0;i+1<n;i++) {
        A[i+1]+=A[i]/10;
        A[i]=A[i]%10;
    }
    while(!A.back() && n>1) A.pop_back(),n--;
    string ret;
    for(int i=0;i<n;i++) {
        ret+=A[n-1-i]+'0';
    }
    return ret;
}
