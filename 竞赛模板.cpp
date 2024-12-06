//-- ICPC竞赛模版
//-- 东莞理工学院 黄海

/////////////////// HEADING ////////////////////

#include <bits/stdc++.h>
using namespace std;
#define int long long 
typedef long double ld;
typedef long long ll;
typedef unsigned long long ull;
#define endl '\n'
#define lowbit(x) ((x)&(-x))

// #include <bits/extc++.h>
// using namespace __gnu_pbds;
// typedef tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update> oset;


void solve() {
    
}

signed main(){
    ios::sync_with_stdio(0); cin.tie(0);

    int _=1;
    cin>>_;
    while(_--) {
        solve();
    }
}


/////////////////// DATA STRUCTURE //////////////////

//-- sparse table

/*
DS for RMQ problems. 'sel' is a function pointer such as 'get_max' or 'get_min'. 'st_queue' returns result from interval [l,r].
*/

template<typename T> vector<vector<T>> get_st(vector<T> &v, T (*sel)(T a, T b)) {
    int n = v.size(), m = log2(n)+1;
    vector<vector<T>> ret(m);
    ret[0] = v;
    for(int i=1;i<m;i++) for(int j=0;j<=n-(1<<i);j++) ret[i].push_back(sel(ret[i-1][j],ret[i-1][j+(1<<(i-1))]));
    return ret;
}

template<typename T> T st_queue(vector<vector<T>> &st, int l, int r, T (*sel)(T a, T b)) {
    int lay = log2(r-l+1), wid=1<<lay;
    return sel(st[lay][l],st[lay][r-wid+1]);
}

//-- DSU without rank

vector<int> dsu(n+1);
iota(dsu.begin(),dsu.end(),0LL);

function<int(int)> find_root = [&](int x) {
    return dsu[x]==x ? x : (dsu[x]=find_root(dsu[x]));
};

function<bool(int,int)> merge = [&](int x, int y) {
    int xr=find_root(x),yr=find_root(y);
    dsu[xr]=yr;
    return xr!=yr;
};

//-- DSU with rank

vector<int> dsu(n),w(n,1);
iota(dsu.begin(),dsu.end(),0LL);

function<int(int)> find_root = [&](int x) {
    return dsu[x]==x ? x : (dsu[x]=find_root(dsu[x]));
};

function<bool(int,int)> merge = [&](int x, int y) {
    int xr=find_root(x),yr=find_root(y);
    dsu[xr]=yr;
    if(xr!=yr) w[yr]+=w[xr];
    return xr!=yr;
};

//-- fenwick tree

/*
Used for single point update(addition only) & range sum.
If it's not addition, try to modify your solution to addition.

USAGE

    Fenwick<int> fen(n+1);
    fen.modify(index, x);
    fen.query(L, R);

The starting index is 1, and intervals are [L,R).
*/

template<typename bas>
struct Fenwick {
    vector<bas> t;
    int n;

    Fenwick(int _n) {
        n=_n;
        t.resize(n);
    }

    void modify(int p, bas x) {
        for(int i=p;i<n;i+=lowbit(i)) t[i]+=x;
    }

    bas sum(int r) {
        bas ret=0;
        for(int i=r;i;i-=lowbit(i)) ret+=t[i];
        return ret;
    }

    bas query(int l, int r) {
        return sum(r-1)-sum(l-1);
    }
};

//-- single-point segment tree

/*

USAGE

This single-point segment tree is multi-purpose, meaning that you have to define your own version
of init/merge/modify as well as information to be maintained. You do this by
passing a struct written by yourself to the segment tree like this:

    vector<int> v(n);
    SegTree<int, S> seg(v);   // S is the struct

And you can use the segment tree like this:

    seg.modify(p, x);
    int result = seg.query(L, R).val;

Notice that 'val' is actually a member element of your 'S' struct. You may replace it with
something else.

The starting index is 0, and intervals are [L,R).
*/

/*
An example of a struct for querying the length of the longest consecutive
all-0 segment within [L, R) on an array of 0/1.

The update is to directly change a[p] to x.

pl is the longest length of all-0 prefix,
sl is the longest length of all-0 suffix,
tl is the answer.
*/

struct S {
    ll pl,sl,tl,len;
    void init(ll x) {
        pl=sl=tl=x==0;
        len=1;
    }

    void merge(S a, S b) {
        len=a.len+b.len;

        if(a.pl==a.len) {
            pl=a.len+b.pl;
        } else {
            pl=a.pl;
        }

        if(b.sl==b.len) {
            sl=b.len+a.sl;
        } else {
            sl=b.sl;
        }

        tl=max(a.tl,b.tl);
        tl=max(tl,a.sl+b.pl);
    }

    void modify(ll x) {
        pl=sl=tl=x==0;
    }
};

/*
single-point multi-purpose segment tree (without lazy propagation)

This one is recursive, which is slightly slower than the one below.
*/

template<typename bas,typename inf>
struct SegTree {
    int n;
    vector<inf> t;
    vector<int> L,R;
    
    SegTree(vector<bas> &v) {
        n=v.size();
        t.resize(4*n);
        L.resize(4*n);
        R.resize(4*n);
        init(v,0,0,n);
    }

    void init(vector<bas> &v, int x, int l, int r) {
        L[x]=l,R[x]=r;
        if(l+1==r) {
            t[x].init(v[l]);
        } else {
            int m=(l+r)/2;
            init(v,x*2+1,l,m);
            init(v,x*2+2,m,r);
            t[x].merge(t[x*2+1],t[x*2+2]);
        }
    }

    inf _query(int x, int l, int r) {
        if(l>=r) return inf();
        if(l==L[x] && r==R[x]) {
            return t[x];
        } else {
            int m=(L[x]+R[x])/2;
            inf a;
            a.merge(_query(x*2+1,l,min(r,m)),_query(x*2+2,max(l,m),r));
            return a;
        }
    }

    inf query(int l, int r) {
        return _query(0,l,r);
    }

    void _modify(int x, int p, bas k) {
        if(L[x]+1==R[x]) {
            t[x].modify(k);
        } else {
            int m=(L[x]+R[x])/2;
            if(p<m) _modify(x*2+1,p,k);
            else _modify(x*2+2,p,k);
            t[x].merge(t[x*2+1],t[x*2+2]);
        }
    }

    void modify(int p, bas k) {
        return _modify(0,p,k);
    }

};

/*
single-point multi-purpose segment tree (without lazy propagation)

This one uses bottom-up implementation so it runs slightly faster than recursive version.
*/

template<typename bas,typename inf>
struct SegTree {
    int n;
    vector<inf> t;

    SegTree(vector<bas> v) {
        n=1<<(int)ceil(log2(v.size()));
        v.resize(n);
        t.resize(2*n);
        for(int i=2*n-1;i;i--) {
            if(i>=n) t[i].init(v[i-n]);
            else t[i].merge(t[i*2],t[i*2+1]);
        }
    }

    inf part(int l, int len) {
        return t[n/len+l/len];
    }

    inf query(int l, int r) {
        inf ans;
        int cnt=0;

        while(l && l+lowbit(l)<=r) {
            inf x=part(l,lowbit(l));
            if(cnt++) ans.merge(ans,x);
            else ans=x;
            l+=lowbit(l);
        }

        for(int i=n;i;i/=2) {
            if(l+i<=r) {
                inf x=part(l,i);
                if(cnt++) ans.merge(ans,x);
                else ans=x;
                l+=i;
            }
        }

        return ans;
    }

    void modify(int p, bas k) {
        t[p+n].modify(k);
        for(int i=(p+n)/2;i;i/=2) t[i].merge(t[i*2],t[i*2+1]);
    }
};

//-- lazy segment tree

/*

USAGE

This lazy segment tree is multi-purpose, meaning that you have to define your own version
of init/merge/modify/pushdown as well as information to be maintained. You do this by
passing a struct written by yourself to the segment tree like this:

    vector<int> v(n);
    LazySegTree<int, S> seg(v);   // S is the struct

And you can use the segment tree like this:

    seg.modify(L, R, x);
    int result = seg.query(L, R).val;

Notice that 'val' is actually a member element of your 'S' struct. You may replace it with
something else.

The starting index is 0, and intervals are [L,R).
*/

/*
An example of a struct for addition & range sum:
*/

struct S {
    ll val,lazy,len=1;
    void init(ll x) {
        val=x;
    }

    void merge(S a, S b) {
        val=a.val+b.val+a.lazy*a.len+b.lazy*b.len;
        len=a.len+b.len;
    }

    void modify(ll x) {
        lazy+=x;
    }

    void pushdown(S &a, S &b) {
        val+=lazy*len;
        a.lazy+=lazy;
        b.lazy+=lazy;
        lazy=0;
    }
};

/*
An example of a struct for modify & range sum:
*/

struct S {
    ll INF=1e18;
    ll val,lazy=INF,len=1;
    void init(ll x) {
        val=x;
    }

    void merge(S a, S b) {
        val=(a.lazy==INF?a.val:(a.lazy*a.len))+(b.lazy==INF?b.val:(b.lazy*b.len));
        len=a.len+b.len;
    }

    void modify(ll x) {
        lazy=x;
    }

    void pushdown(S &a, S &b) {
        if(lazy!=INF) {
            val=lazy*len;
            a.lazy=lazy;
            b.lazy=lazy;
            lazy=INF;
        }
    }
};

/*
An example of a struct for addition & range min & range min count:
*/

struct S {
    ll mini=1e18,cnt,lazy=0;
    void init(ll x) {
        mini=x;
        cnt=1;
    }

    void merge(S a, S b) {
        a.mini+=a.lazy;
        b.mini+=b.lazy;
        if(a.mini==b.mini) {
            mini=a.mini;
            cnt=a.cnt+b.cnt;
        } else if(a.mini<b.mini) {
            mini=a.mini;
            cnt=a.cnt;
        } else {
            mini=b.mini;
            cnt=b.cnt;
        }
    }

    void modify(ll x) {
        lazy+=x;
    }

    void pushdown(S &a, S &b) {
        mini+=lazy;
        a.lazy+=lazy;
        b.lazy+=lazy;
        lazy=0;
    }
};

/*
The multi-purpose lazy segment tree
*/

template<typename bas,typename inf>
struct LazySegTree {
    int n;
    vector<inf> t;
    vector<int> L,R;
    inf tmp;

    LazySegTree(vector<bas> &v) {
        n=v.size();
        t.resize(n*4);
        L.resize(n*4);
        R.resize(n*4);
        init(v,0,0,n);
    }

    void init(vector<int> &v, int x, int l, int r) {
        L[x]=l, R[x]=r;
        if(l+1==r) {
            t[x].init(v[l]);
        } else {
            int m=(l+r)/2;
            init(v,x*2+1,l,m);
            init(v,x*2+2,m,r);
            t[x].merge(t[x*2+1],t[x*2+2]);
        }
    }

    void pushdown(int x) {
        if(L[x]+1==R[x]) {
            t[x].pushdown(tmp,tmp);
        } else {
            t[x].pushdown(t[x*2+1],t[x*2+2]);
        }
    }

    inf _query(int x, int l, int r) {
        if(l>=r) return inf();
        pushdown(x);
        if(l==L[x] && r==R[x]) {
            return t[x];
        } else {
            int m=(L[x]+R[x])/2;
            inf a;
            a.merge(_query(x*2+1,l,min(r,m)), _query(x*2+2,max(l,m),r));
            return a;
        }
    }

    inf query(int l, int r) {
        return _query(0,l,r);
    }

    void _modify(int x, int l, int r, bas k) {
        if(l>=r) return;
        pushdown(x);
        if(l==L[x] && r==R[x]) {
            return t[x].modify(k);
        } else {
            int m=(L[x]+R[x])/2;
            _modify(x*2+1,l,min(r,m),k);
            _modify(x*2+2,max(l,m),r,k);
            t[x].merge(t[x*2+1],t[x*2+2]);
        }
    }

    void modify(int l, int r, bas k) {
        _modify(0,l,r,k);
    }
};

//////////// DP /////////////////

//-- SOS DP

vector<ll> SOS_exset(vector<ll> v) {
    int n=v.size();
    for(int i=0;i<log2(n);i++) {
        for(int j=0;j<n;j++) {
            if((j>>i&1)==0) v[j]+=v[j^(1<<i)];
        }
    }
    return v;
}

vector<ll> SOS_subset(vector<ll> v) {
    int n=v.size();
    for(int i=0;i<log2(n);i++) {
        for(int j=n-1;j>=0;j--) {
            if((j>>i&1)==1) v[j]+=v[j^(1<<i)];
        }
    }
    return v;
}

///////////// TREE //////////////

//-- multi-purpose tree template

/*
USAGE

1. Create a tree and specify the root:

    Tree tree(adj, root);

or 

    Tree tree(adj);

when root=1.

2. Get the children of node u:

    vector<int> children = tree.child[u];

3. Get the parent of node u:

    int parent = tree.acs[0][u];

4. Get the d-th ancestor of node u:

    int ancestor = tree.find_acs(u, d);

5. Get the depth of node u:

    int depth = tree.dep[u];

6. Get the height of node u:

    int height = tree.hei[u];
*/

struct Tree {
    int n,root;
    vector<vector<int>> adj, child, acs;

    vector<int> dep,hei;

    Tree(vector<vector<int>> &adj_, int root_=1) {
        adj=adj_;
        root=root_;
        n=adj.size()-1;
        child=vector<vector<int>>(n+1);
        dep=hei=vector<int>(n+1);
        acs=vector<vector<int>>(20,vector<int>(n+1));
        dfs(root,0);
        for(int i=1;i<20;i++) {
            for(int j=1;j<=n;j++) {
                acs[i][j]=acs[i-1][acs[i-1][j]];
            }
        }
    }

    void dfs(int curr, int fa) {
        acs[0][curr]=fa;
        dep[curr]=dep[fa]+1;
        for(auto e:adj[curr]) {
            if(e==fa) continue;
            child[curr].push_back(e);
            dfs(e,curr);
            hei[curr]=max(hei[curr],hei[e]+1);
        }
    }

    int find_acs(int u, int d) {
        while(d) {
            int lb=d&-d;
            u=acs[__lg(lb)][u];
            d-=lb;
        }
        return u;
    }

    int find_lca(int u, int v) {
        if(dep[u]<dep[v]) swap(u,v);
        u=find_acs(u,dep[u]-dep[v]);

        if(u==v) return u;

        for(int i=19;i>=0;i--) {
            int ua=acs[i][u], va=acs[i][v];
            if(ua!=va) u=ua,v=va;
        }
        
        return acs[0][u];
    }

    pair<int,int> find_lca_dis(int u, int v) {
        bool flag=false;
        if(dep[u]<dep[v]) {
            swap(u,v);
            flag=true;
        }
        int dd=dep[u]-dep[v];
        u=find_acs(u,dd);

        pair<int,int> ans;

        if(u==v) {

            ans={dd,0};

        } else {

            int cnt=0;

            for(int i=19;i>=0;i--) {
                int ua=acs[i][u], va=acs[i][v];
                if(ua!=va) {
                    u=ua,v=va;
                    cnt+=(1LL<<i);
                }
            }
            
            ans={cnt+dd+1,cnt+1};
        }

        if(flag) swap(ans.first,ans.second);
        
        return ans;
    }
};

///////////// GRAPH /////////////

//-- dijkstra

/*
Note that the data type of weight is 'dt'. You may change 'typedef ll dt' as you want.
*/

typedef ll dt;
vector<dt> dijkstra(vector<vector<pair<int,dt>>> &adj, int s) {
    vector<dt> dp(adj.size(),1e18); dp[s]=0;
    vector<int> vis(adj.size());
    priority_queue<pair<dt,int>> pq; pq.emplace(0,s);
    while(pq.size()) {
        int curr=pq.top().second; pq.pop();
        if(vis[curr]++) continue;
        for(auto [e,w]:adj[curr]) {
            dp[e]=min(dp[e],dp[curr]+w);
            pq.emplace(-dp[e],e);
        }
    }
    return dp;
}


//-- bellman ford

/*
Note that the data type of weight is 'dt'. You may change 'typedef ll dt' as you want.
*/

typedef ll dt;
ld bellman_ford(vector<vector<pair<int,ld>>> &adj,int s,int t) {
    int n=adj.size();
    vector<ld> dp(n,1e12);
    dp[s]=0;
    for(int i=0;i<n-1;i++) {
        for(int j=0;j<n;j++) {
            for(auto [nv,w]:adj[j]) {
                dp[nv]=min(dp[nv],dp[j]+w);
            }
        }
    }
    return dp[t];
}

//-- kosaraju

/*
Algorithm for finding SCC
*/

vector<vector<int>> kosaraju(vector<vector<int>> &adj) {
    int n=adj.size();
    vector<vector<int>> scc,g_(n);
    vector<int> vis(n),ord;

    for(int i=0;i<n;i++) {
        for(auto j:adj[i]) g_[j].push_back(i);
    }
    
    function<void(int)> dfs1=[&](int curr) {
        vis[curr]=1;
        for(auto e:g_[curr]) {
            if(!vis[e]) dfs1(e);
        }
        ord.push_back(curr);
    };

    function<void(int)> dfs2=[&](int curr) {
        vis[curr]=1;
        for(auto e:adj[curr]) {
            if(!vis[e]) dfs2(e);
        }
        scc.back().push_back(curr);
    };

    for(int i=0;i<n;i++) if(!vis[i]) dfs1(i);
	
	vis=vector<int>(n);

    for(int i=n-1;i>=0;i--) {
        if(!vis[ord[i]]) {
            scc.push_back({});
            dfs2(ord[i]);
        }
    }

    return scc;
}

//-- tarjan for SCC

vector<vector<int>> tarjan_SCC(vector<vector<int>> &adj) {

	int n=adj.size(),ts=0;
	vector<vector<int>> scc;
	vector<int> low(n,1e9),ins(n);
	stack<int> st;

	function<void(int)> dfs=[&](int curr) {

		st.push(curr);
		low[curr]=++ts;
		int ots=ts;
		ins[curr]=1;

		for(auto e:adj[curr]) {
			if(low[e]==1e9) dfs(e);
			if(ins[e]) {
				low[curr]=min(low[curr],low[e]);
			}
		}

		if(low[curr]==ots) {
			scc.push_back({});
			do {
				scc.back().push_back(st.top());
				ins[st.top()]=0;
				st.pop();
			} while(scc.back().back()!=curr);
		}	
	};

	for(int i=0;i<n;i++) {
		if(low[i]==1e9) dfs(i);
	}

	return scc;
}

//-- kruskal for MST

/*
'edge' format {weight, vertex1, vertex2}
*/

vector<vector<int>> kruskal_MST(int n, vector<vector<int>> &edge) {

    vector<int> dsu(n+1);
    iota(dsu.begin(),dsu.end(),0LL);

    function<int(int)> find_root = [&](int x) {
        return dsu[x]==x ? x : (dsu[x]=find_root(dsu[x]));
    };

    function<bool(int,int)> merge = [&](int x, int y) {
        int xr=find_root(x),yr=find_root(y);
        dsu[xr]=yr;
        return xr!=yr;
    };

    sort(edge.begin(),edge.end());

    vector<vector<int>> ret;

    for(auto e:edge) {
        int w=e[0],u=e[1],v=e[2];
        if(find_root(u)!=find_root(v)) {
            ret.push_back(e);
            merge(u,v);
        }
    }

    return ret;
}

////////////////////////// STRING //////////////////////////

//-- kmp

vector<int> kmp_next(string t) {
	
	int n=t.size()+1;
	vector<int> ne(n);

	for(int i=2;i<n;i++) {
		int j=ne[i-1];
		while(t[i-1]!=t[j] && j) j=ne[j];
		j+=t[i-1]==t[j];
		ne[i]=j;
	}

	return ne;
}

int kmp_first(string s, string t) {

	auto ne=kmp_next(t);
	int i=0,j=0,n=s.size(),m=t.size();

	for(int i=0;i<n;i++) {
		while(s[i]!=t[j] && j) j=ne[j];
		if(s[i]==t[j]) j++;
		if(j==m) {
			return i-m+1;
		}
	}

	return -1;
}

vector<int> kmp_all(string s, string t) {
	
	auto ne=kmp_next(t);
	vector<int> ans;
	int i=0,j=0,n=s.size(),m=t.size();

	for(int i=0;i<n;i++) {
		while(s[i]!=t[j] && j) j=ne[j];
		if(s[i]==t[j]) j++;
		if(j==m) {
			j=ne[j];
			ans.push_back(i-m+1);
		}
	}
	
	return ans;
}

/////////////////////// MATH /////////////////////////

//-- factorization

ll MOD;
vector<ll> factorize(ll x) {
    vector<ll> ret;
    for(ll i=2;i*i<=x;i++) {
        if(x%i==0) {
            ret.push_back(i);
            x/=i;
            i=1;
        }
    }
    if(x!=1) ret.push_back(x);
    return ret;
}

//-- fast factorization

vector<int> primes;
vector<ll> factorize(ll x) {
    vector<ll> ret;
    for(int i=0;i<primes.size();i++) {
        if(x%primes[i]==0) {
            ret.push_back(primes[i]);
            x/=primes[i];
            i=-1;
        }
    }
    if(x!=1) ret.push_back(x);
    return ret;
}

//-- count integers in [1,m] coprime to every v[i]

ll coprime_count(ll m,vector<ll> &v) {
    int n=v.size();
    ll ret=0;
    for(ll i=1;i<(1<<n);i++) {
        ll base=1,cnt=-1;
        for(ll j=0;j<n;j++) {
            if(i&(1LL<<j)) {
                base*=v[j];
                cnt*=-1;
            }
        }
        ret+=m/base*cnt;
    }
    return m-ret;
}

//-- quick exponentiation

ll qpow(ll a, ll n) {
	ll ret=1;
	while(n) {
		if(n%2) ret=ret*a%MOD;
		a=a*a%MOD;
		n/=2;
	}
	return ret;
}

//-- quick factorial

vector<ll> F,iF;
void initF(int n) {
    F=iF=vector<ll>(n+1);
    F[0]=1;
    for(int i=1;i<=n;i++) {
        F[i]=F[i-1]*i%MOD;
    }
    iF[n]=qpow(F[n],MOD-2);
    for(int i=n-1;i>=0;i--) {
        iF[i]=iF[i+1]*(i+1)%MOD;
    }
}

//-- number of ways to unorderly select m out of n modulo MOD

ll C(ll n, ll m) {
	return n<m?0:F[n]*iF[m]%MOD*iF[n-m]%MOD;
}


//-- number of ways to orderly select m out of n modulo MOD

ll A(ll n, ll m) {
    return n<m?0:F[n]*iF[n-m]%MOD;
}

//-- pre-calculate powers of 2

vector<ll> p2;
void initp2(int n) {
    p2=vector<ll>(n+1);
    p2[0]=1;
    for(int i=1;i<=n;i++) p2[i]=p2[i-1]*2%MOD;
}

//-- miller rabin primality test

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

//-- linear sieve for primes

vector<int> linear_sieve(int n) {
    vector<int> primes,mp(n+1);
    for(int i=2;i<=n;i++) {
        if(!mp[i]) {
            primes.push_back(i);
            mp[i]=i;
        }
        for(auto p:primes) {
            if(i*p>n) break;
            mp[i*p]=p;
            if(i%p==0) break;
        }
    }
    return primes;
}

//-- linear sieve for smallest prime factor

vector<int> linear_sieve(int n) {
    vector<int> primes,mp(n+1);
    for(int i=2;i<=n;i++) {
        if(!mp[i]) primes.push_back(i);
        for(auto p:primes) {
            if(i*p>n) break;
            mp[i*p]=p;
            if(i%p==0) break;
        }
    }
    return mp;
}

//-- extended euclidean

ll exgcd(ll a,ll b,ll &x,ll &y) {
    if(!b) {
        x=1,y=0;
        return a;
    }
    ll ans=exgcd(b,a%b,x,y);
    x-=a/b*y;
    swap(x,y);
    return ans;
}

//-- find inverse modulo non-prime

ll mod_inverse(ll a,ll mod) {
    ll x,y;
    ll d=exgcd(a,mod,x,y);
    return d==1?(x%mod+mod)%mod:-1;
}

//-- fast way to count inversions in an array

int mergeSort(vector<int>& record, vector<int>& tmp, int l, int r) {
    if (l >= r) {
        return 0;
    }

    int mid = (l + r) / 2;
    int inv_count = mergeSort(record, tmp, l, mid) + mergeSort(record, tmp, mid + 1, r);
    int i = l, j = mid + 1, pos = l;
    while (i <= mid && j <= r) {
        if (record[i] <= record[j]) {
            tmp[pos] = record[i];
            ++i;
            inv_count += (j - (mid + 1));
        }
        else {
            tmp[pos] = record[j];
            ++j;
        }
        ++pos;
    }
    for (int k = i; k <= mid; ++k) {
        tmp[pos++] = record[k];
        inv_count += (j - (mid + 1));
    }
    for (int k = j; k <= r; ++k) {
        tmp[pos++] = record[k];
    }
    copy(tmp.begin() + l, tmp.begin() + r + 1, record.begin() + l);
    return inv_count;
}

int count_inversions(vector<int> record) {
    int n = record.size();
    vector<int> tmp(n);
    return mergeSort(record, tmp, 0, n - 1);
}

//-- FFT

// complex number
struct CN {
    ld a=0,b=0;
};

// complex addition
CN operator+(CN a, CN b) {
    a.a+=b.a, a.b+=b.b;
    return a;
}

// complex multiplication
CN operator*(CN a, CN b) {
    CN c {a.a*b.a-a.b*b.b, a.a*b.b+a.b*b.a};
    return c;
}

// FFT for a polynomimal
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

// inverse FFT for a polynomial
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

// polynomial multiplication
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
        a[i]=A[i].a+0.1;
    }
    while(!a.back() && a.size()>1) a.pop_back();
    return a;
}

// large integer multiplication
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

//-- floor sum
ll floor_sum(ll a, ll b, ll c, ll n) {
    if (a == 0) return (n + 1) * (b / c);
    ll Ta = n * (n + 1) / 2, Tb = n + 1;
    if (a >= c || b >= c)
        return (floor_sum(a % c, b % c, c, n) + (a / c) * Ta + (b / c) * Tb);
    ll d = (a * n + b) / c;
    return (n + 1) * d - floor_sum(c, c - b + a - 1, a, d - 1);
}
