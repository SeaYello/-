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
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

//////// DATA STRUCTURE ////////

int log2d(ll x) {
    int ret=0;
    while(x>>=1) ret++;
    return ret;
}

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

/////// MATH /////////

ll qpow(ll a, ll n, ll p) {
	ll ret=1;
	while(n) {
		if(n%2) ret=ret*a%p;
		a=a*a%p;
		n/=2;
	}
	return ret;
}

ll C(ll n, ll m, ll p) {
	ll a=1,b=1;
	for(int i=1;i<=m;i++) {
		a=a*(n-i+1)%p;
		b=b*i%p;
	}
	return a*qpow(b,p-2,p)%p;
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
		ll x=qpow(a,d,n);
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

////////////////////////////
