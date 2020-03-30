type SetTransform<T> = (transform: Array<T>) => Array<T>;
type NodeType = 'root' | 'node';
type Source<T> = string | InfiniNode<T>;
type Reducer<T> = (node: InfiniNode<T>, acc?: any) => any;
type Stringify = (t?: any) => string;

export type NodeParams<T> = {
  type?: NodeType;
  source?: Source<T>;
  children?: Array<InfiniNode<T>>;
  transform?: Array<T>;
  reducer?: Reducer<T>;
  stringify?: Stringify;
};

export class InfiniCollection<T> {
  reducer: Reducer<T>;
  stringify: Stringify;
  children: Array<InfiniNode<T>> = [];
  constructor(params: {reducer: Reducer<T>; stringify: Stringify}) {
    this.reducer = params.reducer;
    this.stringify = params.stringify;
  }
  create(source: string, transform?: T[]) {
    const node = new InfiniNode({
      type: 'root',
      source: source,
      transform: transform,
      reducer: this.reducer,
      stringify: this.stringify,
    });
    this.appendChild(node);
    return node;
  }
  appendChild(child: InfiniNode<T>) {
    this.children.push(child);
    return child;
  }
  setTransform(setter: SetTransform<T>): Array<Array<T>> {
    this.children.forEach(child => child.setTransform(setter));
    return this.children.map(child => child.transform);
  }
}

export class InfiniNode<T> {
  public type: NodeType;
  public source: Source<T>;
  public reducer: Reducer<T>;
  public stringify: Stringify;
  public children: Array<InfiniNode<T>>;
  protected _transform: Array<T>;

  setTransform(setter: SetTransform<T>) {
    this.transform = setter(this.transform);
  }

  set transform(transform: Array<T>) {
    this._transform = transform;
  }

  get transform(): Array<T> {
    return this._transform;
  }

  constructor(params: NodeParams<T>) {
    this._transform = params.transform || [];
    this.type = params.type || 'node';
    this.source = params.source!;
    this.children = params.children || [];
    this.reducer = params.reducer!;
    this.stringify = params.stringify!;
  }

  add(params: NodeParams<T>) {
    const child = new InfiniNode<T>({
      type: 'node',
      source: this,
      transform: params.transform || [],
      children: params.children || [],
      reducer: this.reducer,
      stringify: this.stringify,
    });
    this.appendChild(child);
    return child;
  }

  appendChild(child: InfiniNode<T>) {
    this.children.push(child);
    child.source = this;
    return child;
  }

  reduce(node: InfiniNode<T> = this, acc?: any): any {
    if (this.type !== 'root') {
      const source = this.source as InfiniNode<T>;
      const _acc = this.reducer(this, acc);
      return source.reduce(node, _acc);
    }
    return this.reducer(this, acc);
  }

  reduceToString() {
    return this.stringify(this.reduce());
  }
}
