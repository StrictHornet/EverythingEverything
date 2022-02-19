class Hashtable {
    constructor(){
        this.table = new Array(127);
        this.size = 0;
    }

    _hash(key) {
        let hash = 0;
        for(let i = 0; i < key.length; i++){
            hash += key.charCodeAt(i);
        }

        return hash % this.table.length;
    }
    
    set(key, value){
        let index = this._hash(key);
        this.table[index] = [key, value]
        this.size++
    }

    get(key){
        let index = this._hash(key)
        return this.table[index]
    }

    remove(key){
        let index = this._hash(key)

        if(this.table[index]){
            this.table[index] = []
            this.size -= 1
            return true
        }
        return false
    }
}

const ht = new HashTable();
ht.set("Canada", 300);
ht.set("France", 100);
ht.set("Spain", 110);

console.log(ht.get("Canada")); // [ 'Canada', 300 ]
console.log(ht.get("France")); // [ 'France', 100 ]
console.log(ht.get("Spain")); // [ 'Spain', 110 ]