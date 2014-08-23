var db = Titanium.Database.open('recommender.db');
// needs more work!! since it can create a new database there. Need to set the right directory!!!!

// retrieve the product names part
var product_name = 'product_id';
var table_name = 'yahoo_data';
var sql_command = "SELECT " + product_name + "FROM" + table_name;
var products_table = db.execute(sql_command);

var products =[]
while(products_table.isValidRow()){
	var aProduct = products_table.fieldByName('prodcut_id');
	products.push(aProduct)
}