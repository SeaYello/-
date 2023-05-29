#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
#define rep(x,y,z) for(ll x=y;x<=z;x++)

template<typename T> T bisel(bool (*f)(T x), T l, T r) {
    while(r-l>1) {
        ll m=(l+r)/2;
        (f(l)==f(m)?l:r) = m;
    } return l;
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
    rep(i,0,(int)w.size()-1) {
        for(int j=tv;j>=0;j--) {
            ll a = dp[j];
            ll b = (j-w[i])<0?0:(dp[j-w[i]]+v[i]);
            dp[j]=max(a,b);
        }
    }
    return dp.back();
}

vector<int> string_search(string s, string p) {
	ull h1=0,h2=0,b=3242343,bpow=1;
	vector<int> ret;
	int n=s.size(), m=p.size();
	s+='x';
	for(int i=m-1;i>=0;i--) {
		h1+=bpow*s[i];
		h2+=bpow*p[i];
		bpow*=b;
	}
	for(int i=0;i<n-m+1;i++) {
		if(h1==h2) ret.push_back(i);
		h1=h1*b+s[i+m];
		h1-=s[i]*bpow;
	}
	return ret;
}

ll qpow(ll a, ll n, ll p) {
	ll ret=1;
	for(int i=0;i<30;i++) {
		if(((n&(1<<i))>>i)) ret=ret*a%p;
		a=a*a%p;
	}
	return ret;
}

ll inv(ll a, ll p) {
	return qpow(a,p-2,p);
}

ll C(ll n, ll m, ll p) {
	ll a=1,b=1;
	for(int i=1;i<=m;i++) {
		a=a*(n-i+1)%p;
		b=b*i%p;
	}

	return a*inv(b,p)%p;
}

/////////////////////////////////////////////////////
